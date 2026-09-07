"""
Extract actual token usage from experiment log and calculate real cost.
"""
import json
import re
from pathlib import Path

# ============ 价格配置 ============
# 用哪个模型就改成对应的价格，单位是 per 1K tokens
#
# gpt-4o (USD):      输入 0.0025   输出 0.01     ($2.50/1M, $10/1M)
# gpt-4o-mini (USD): 输入 0.00015  输出 0.0006   ($0.15/1M, $0.60/1M)
# Qwen3.5-9B (CNY): 输入 0.0005   输出 0.004    (¥0.5/M, ¥4/M)
#
# 切换模型时改下面三行 + 第22行的 log_path + CURRENCY
# ====================================================
PRICE_PROMPT_PER_1K = 0.0005      # 当前: Qwen3.5-9B
PRICE_COMPLETION_PER_1K = 0.004   # 当前: Qwen3.5-9B
CURRENCY = "CNY"

log_path = Path(__file__).parent / "logs" / "llm4plc_exp_20260428_163145.log"

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Extract all API usage records (handles nested dicts like prompt_tokens_details)
pattern = r"LLM API call successful: (\{.*\})"
matches = re.findall(pattern, content)

total_prompt = 0
total_completion = 0
total_all = 0
phase1 = {"prompt": 0, "completion": 0, "count": 0}
phase2 = {"prompt": 0, "completion": 0, "count": 0}
phase3 = {"prompt": 0, "completion": 0, "count": 0}

# Track by phase using surrounding context
lines = content.split('\n')
current_phase = "unknown"

for line in lines:
    if "Planning Phase" in line or "Phase1" in line:
        current_phase = "phase1"
    elif "Phase2_Generation" in line:
        current_phase = "phase2"
    elif "Phase3_Refinement" in line:
        current_phase = "phase3"

    if "LLM API call successful" in line:
        try:
            # Extract the usage dict using regex (handles nested dicts)
            m = re.search(pattern, line)
            if not m:
                continue
            dict_str = m.group(1)
            # Convert single-quote Python dict to valid JSON and parse
            # Replace single quotes with double quotes for JSON parsing
            # (skip the nested dict values to avoid breaking them)
            import ast
            usage = ast.literal_eval(dict_str)

            prompt = usage.get('prompt_tokens', 0)
            completion = usage.get('completion_tokens', 0)
            total = usage.get('total_tokens', 0)

            total_prompt += prompt
            total_completion += completion
            total_all += total

            if current_phase == "phase1":
                phase1["prompt"] += prompt
                phase1["completion"] += completion
                phase1["count"] += 1
            elif current_phase == "phase2":
                phase2["prompt"] += prompt
                phase2["completion"] += completion
                phase2["count"] += 1
            elif current_phase == "phase3":
                phase3["prompt"] += prompt
                phase3["completion"] += completion
                phase3["count"] += 1
        except Exception as e:
            pass

# Calculate costs if prices are set
cost_prompt = total_prompt / 1000 * PRICE_PROMPT_PER_1K if PRICE_PROMPT_PER_1K > 0 else 0
cost_completion = total_completion / 1000 * PRICE_COMPLETION_PER_1K if PRICE_COMPLETION_PER_1K > 0 else 0
total_cost = cost_prompt + cost_completion

print("=" * 60)
print("  Actual Token Usage from API Responses")
print("=" * 60)

print(f"\nTotal API calls: {len(matches)}")
print(f"\n{'Phase':<15} {'API Calls':>10} {'Prompt Tokens':>15} {'Completion Tokens':>20} {'Total':>10}")
print("-" * 70)
print(f"{'Phase 1 (Plan)':<15} {phase1['count']:>10} {phase1['prompt']:>15,} {phase1['completion']:>20,} {phase1['prompt']+phase1['completion']:>10,}")
print(f"{'Phase 2 (Gen)':<15} {phase2['count']:>10} {phase2['prompt']:>15,} {phase2['completion']:>20,} {phase2['prompt']+phase2['completion']:>10,}")
print(f"{'Phase 3 (Refine)':<15} {phase3['count']:>10} {phase3['prompt']:>15,} {phase3['completion']:>20,} {phase3['prompt']+phase3['completion']:>10,}")
print("-" * 70)
print(f"{'TOTAL':<15} {len(matches):>10} {total_prompt:>15,} {total_completion:>20,} {total_all:>10,}")

# Compare with what the script estimated (from summary JSON)
summary_path = Path(__file__).parent / "results" / "gpt-4o_20260428_163145" / "llm4plc_baseline_20260428_163145_summary.json"
if summary_path.exists():
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    est = summary.get('experiment_summary', {}).get('total_tokens', {})
    est_total = est.get('total', 0)
    print(f"\nScript estimated tokens (only Phase 2): {est_total:,}")
    print(f"Actual total tokens (all phases):      {total_all:,}")
    print(f"Difference:  actual is {total_all / est_total:.1f}x the estimate")

print(f"\n--- Cost Calculation ---")
print(f"Pricing: Prompt={PRICE_PROMPT_PER_1K}/{CURRENCY} per 1K, Completion={PRICE_COMPLETION_PER_1K}/{CURRENCY} per 1K")
if PRICE_PROMPT_PER_1K > 0:
    print(f"\nEstimated Cost: {total_cost:.6f} {CURRENCY}")
    print(f"  - Prompt cost:     {total_prompt:,} tokens × {PRICE_PROMPT_PER_1K}/1K = {cost_prompt:.6f} {CURRENCY}")
    print(f"  - Completion cost: {total_completion:,} tokens × {PRICE_COMPLETION_PER_1K}/1K = {cost_completion:.6f} {CURRENCY}")
else:
    print("\n[!] Fill in PRICE_PROMPT_PER_1K and PRICE_COMPLETION_PER_1K in the script to calculate cost.")
    print(f"  Then run: python {Path(__file__).name}")
