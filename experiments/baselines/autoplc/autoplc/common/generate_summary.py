"""
Post-experiment summary generator for AutoPLC.

Creates llm4plc_baseline-compatible JSON output files:
  - {model}_{timestamp}.json          (per-task results array)
  - {model}_{timestamp}_summary.json   (aggregated statistics)

Usage from experiment runner:
    from common.generate_summary import generate_experiment_summary
    generate_experiment_summary(
        base_folder=base_folder,
        benchmark_file=benchmark_file_path,
        config=config,
        statistics_file=stats_path
    )
"""
import json
import os
import re
import glob
import sys
from datetime import datetime
from collections import defaultdict

# ── Pricing from .env (fallback defaults) ──────────────────────────────────
# Overridden by PRICE_PROMPT_PER_1K / PRICE_COMPLETION_PER_1K in .env
DEFAULT_PRICES = {
    "gpt-4o":                {"prompt": 0.0025, "completion": 0.01,   "currency": "USD"},
    "gpt-4o-mini":           {"prompt": 0.00015,"completion": 0.0006, "currency": "USD"},
    "DeepSeek-V3.2":         {"prompt": 0.002,  "completion": 0.003,  "currency": "CNY"},
    "deepseek-ai/DeepSeek-V3.2": {"prompt": 0.002, "completion": 0.003, "currency": "CNY"},
    "Qwen3.5-9B":           {"prompt": 0.0005, "completion": 0.004,  "currency": "CNY"},
    "Qwen/Qwen3.5-9B":       {"prompt": 0.0005, "completion": 0.004,  "currency": "CNY"},
    "qwen3.5-9b":            {"prompt": 0.0005, "completion": 0.004,  "currency": "CNY"},
    "claude-3-5-sonnet-20241022": {"prompt": 0.003, "completion": 0.015, "currency": "USD"},
    "claude-sonnet-4-6":     {"prompt": 0.003, "completion": 0.015, "currency": "USD"},
    "claude-opus-4-7":       {"prompt": 0.015, "completion": 0.075, "currency": "USD"},
}


def _get_pricing(model_name: str):
    """Get pricing for a given model name, checking .env overrides first."""
    # 优先使用模型已知定价表（不受当前 .env 影响）
    model_key = model_name
    if model_name not in DEFAULT_PRICES:
        for key in DEFAULT_PRICES:
            if key in model_name or model_name in key:
                model_key = key
                break
        else:
            model_key = None

    if model_key:
        info = DEFAULT_PRICES[model_key]
        # .env 中的价格仅作为模型不在定价表时的补充
        price_prompt = float(os.environ.get("PRICE_PROMPT_PER_1K", "0") or "0")
        price_comp = float(os.environ.get("PRICE_COMPLETION_PER_1K", "0") or "0")
        currency = os.environ.get("CURRENCY", info["currency"])
        if price_prompt > 0 and price_comp > 0 and model_key is None:
            return price_prompt, price_comp, currency
        return info["prompt"], info["completion"], info["currency"]

    # 完全未知的模型，从 .env 读取
    price_prompt = float(os.environ.get("PRICE_PROMPT_PER_1K", "0.002") or "0.002")
    price_comp = float(os.environ.get("PRICE_COMPLETION_PER_1K", "0.003") or "0.003")
    currency = os.environ.get("CURRENCY", "CNY")
    return price_prompt, price_comp, currency


# 全局缓存 CompilerFrontend 实例（避免每任务重复加载）
_ANTLR_COMPILER = None


def _antlr_validate(st_code: str, project_root: str = None):
    """
    复用项目原版的 CompilerFrontend.scl_syntax_check() 进行 ANTLR 验证。
    返回 (passed: bool, error_message: str)。
    project_root 参数保留仅用于兼容调用方，实际未使用。
    """
    global _ANTLR_COMPILER
    if _ANTLR_COMPILER is None:
        sys.path_importer_cache.clear()
        src_dir = os.path.join(os.path.dirname(__file__), "..")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from autoplc_st.tools.antlr import CompilerFrontend
        _ANTLR_COMPILER = CompilerFrontend("st")

    if not st_code:
        return False, "Empty code"

    # 预处理：给块结束关键字补分号（CODESYS ST 不强制要求，但 SCL 语法解析器需要）
    st_code = re.sub(
        r'(END_(?:IF|FOR|WHILE|CASE|REPEAT))\s*\n',
        r'\1;\n',
        st_code
    )

    try:
        error_log, passed = _ANTLR_COMPILER.scl_syntax_check(st_code)
        if passed:
            return True, "ST output passed ANTLR syntax validation"
        else:
            # 截取前 500 字符的错误信息
            return False, error_log[:500] if error_log else "ANTLR syntax check failed"
    except Exception as e:
        return False, f"ANTLR parse exception: {e}"


def _load_statistics(statistics_file: str):
    """Load token usage statistics from the experiment log."""
    if not os.path.exists(statistics_file):
        return {}

    try:
        with open(statistics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def generate_experiment_summary(
    base_folder: str,
    benchmark_file: str,
    config,
    statistics_file: str = None,
    method: str = "autoplc",
    model_name: str = None,
):
    """
    Main entry point. Generates per-task JSON + summary JSON.

    Iteration data来源: AutoDebugger 在实验中已通过 CODESYS 编译-修复循环产生
    verify_info.jsonl, 每行对应一轮迭代:
      - st_before_fix: 该轮送入 CODESYS 的代码 (iteration_N 的初始代码)
      - compiler: CODESYS 编译结果 (空=通过, 有内容=失败)
      - assistant: LLM 修复方案

    仅当 experiment 阶段 AutoDebugger 连上 CODESYS 时, verify_info 才会存在。
    """
    import logging
    logger = logging.getLogger(method)

    model_name = model_name or config.model
    price_prompt, price_comp, currency = _get_pricing(model_name)

    # Resolve root path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    if statistics_file is None:
        log_dir = getattr(config, "LOG_DIR", os.path.join(project_root, "logs"))
        exp_folder = os.path.basename(base_folder)
        statistics_file = os.path.join(
            str(log_dir), exp_folder, "llm", "statistics.json"
        )

    # Load dataset
    dataset = {}
    if os.path.exists(benchmark_file):
        with open(benchmark_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    dataset[obj.get("name", obj.get("task_id", ""))] = obj
        logger.info(f"Loaded {len(dataset)} tasks from {benchmark_file}")
    else:
        logger.warning(f"Benchmark file not found: {benchmark_file}")

    # Load statistics
    stats_data = _load_statistics(statistics_file)

    # Load timing data
    experiment_total_time = 0.0
    per_task_time = {}
    timing_file = os.path.join(base_folder, "_experiment_timing.json")
    if os.path.exists(timing_file):
        try:
            with open(timing_file, "r", encoding="utf-8") as f:
                timing_data = json.load(f)
            experiment_total_time = timing_data.get("total_execution_time", 0.0)
            per_task_time = timing_data.get("task_timings", {})
            logger.info(f"Loaded timing: total={experiment_total_time}s, {len(per_task_time)} tasks")
        except Exception:
            pass
    else:
        logger.warning(f"Timing file not found: {timing_file}")

    # Gather all task directories (exclude non-task dirs like config)
    exclude_dirs = {"config", "results", "llm4plc_format"}
    task_dirs = sorted([
        d for d in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, d)) and d not in exclude_dirs
    ])

    logger.info(f"Found {len(task_dirs)} task dirs in {base_folder}")

    results = []
    cat_stats = defaultdict(lambda: {
        "total": 0, "solved": 0, "antlr_solved": 0,
        "codesys_solved": 0, "both_solved": 0, "cost": 0.0, "time": 0.0
    })

    # 迭代统计: iteration_0 ~ iteration_3
    iteration_keys = ["iteration_0", "iteration_1", "iteration_2", "iteration_3"]
    iter_stats = {k: {"total": 0, "antlr_passed": 0, "codesys_passed": 0, "both_passed": 0} for k in iteration_keys}

    for name in task_dirs:
        task_dir = os.path.join(base_folder, name)

        # Get task info from dataset
        task_info = dataset.get(name, {})
        description = task_info.get("description", "") or task_info.get("prompt", "") or ""
        category = task_info.get("category", "Unknown")

        # --- Read verify_info.jsonl (AutoDebugger 的实验迭代记录) ---
        verify_path = os.path.join(task_dir, "verify_info.jsonl")
        verify_rounds = []
        codesys_ever_available = False
        if os.path.exists(verify_path):
            try:
                with open(verify_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            verify_rounds.append(json.loads(line))
                codesys_ever_available = len(verify_rounds) > 0
            except Exception:
                pass

        # --- 读取生成的 .st 文件 ---
        st_files = sorted(glob.glob(os.path.join(task_dir, f"{name}_*.st")))
        generated_code = ""
        if st_files:
            try:
                with open(st_files[-1], "r", encoding="utf-8") as f:
                    generated_code = f.read().strip()
            except Exception:
                pass

        # --- 构建每轮迭代的验证数据 ---
        # verify_info 每行对应一轮:
        #   line i → iteration_i
        #   st_before_fix = iteration_i 送 CODESYS 前的代码
        #   compiler 空 = CODESYS 通过
        # 最终 .st 文件中的代码才是 LLM 最后一轮修复后的结果,
        # 因此最终的 ANTLR 用 .st 文件判定, 而非最后一个 st_before_fix。

        task_iterations = []  # [(antlr_pass, codesys_pass), ...]

        if verify_rounds:
            for i, vround in enumerate(verify_rounds):
                code = vround.get("st_before_fix", "")
                compiler_errors = vround.get("compiler", [])
                codesys_pass = len(compiler_errors) == 0
                antlr_pass, _ = _antlr_validate(code, project_root) if code else (False, "")
                task_iterations.append((antlr_pass, codesys_pass))
            # 最终结果: 用 .st 文件的 ANTLR + 最后一轮 CODESYS 结果
            final_cs_pass = task_iterations[-1][1] if task_iterations else False
            final_antlr_pass, _ = _antlr_validate(generated_code, project_root) if generated_code else (False, "")
            task_iterations.append((final_antlr_pass, final_cs_pass))
        else:
            # 没有 verify_info: CODESYS 不可用, 只有迭代0
            codesys_ever_available = False
            antlr_pass, _ = _antlr_validate(generated_code, project_root) if generated_code else (False, "")
            task_iterations.append((antlr_pass, False))

        # --- 填充 iteration_stats ---
        for i, (antlr_p, codesys_p) in enumerate(task_iterations):
            if i >= len(iteration_keys):
                break
            key = iteration_keys[i]
            iter_stats[key]["total"] += 1
            if antlr_p:
                iter_stats[key]["antlr_passed"] += 1
            if codesys_p:
                iter_stats[key]["codesys_passed"] += 1
            if antlr_p and codesys_p:
                iter_stats[key]["both_passed"] += 1

        # --- 最终态 (取最后一轮的结果) ---
        final_antlr, final_codesys = task_iterations[-1] if task_iterations else (False, False)

        # Token counts from statistics
        prompt_tok = 0
        completion_tok = 0
        if stats_data:
            task_stats = stats_data.get("tasks", {}).get(name, {})
            if task_stats:
                task_total = task_stats.get("task_total", {})
                prompt_tok = task_total.get("input_tokens", 0)
                completion_tok = task_total.get("output_tokens", 0)

        # Fallback: estimate from code size if stats show placeholder values (< 50 tokens)
        if prompt_tok + completion_tok < 50 and generated_code:
            import tiktoken
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                prompt_tok = len(enc.encode(description)) if description else len(description) // 2
                completion_tok = len(enc.encode(generated_code))
            except Exception:
                prompt_tok = max(1, len(description) // 2)
                completion_tok = max(1, len(generated_code) // 2)

        total_tok = prompt_tok + completion_tok
        cost = (prompt_tok / 1000 * price_prompt) + (completion_tok / 1000 * price_comp)

        is_solved = final_antlr and final_codesys
        generation_success = bool(generated_code)
        generation_error = None if generation_success else "No .st file generated"

        codesys_errors = []
        if not codesys_ever_available:
            codesys_errors = [{"ErrorDesc": "CODESYS compiler service not available"}]
        elif not final_codesys and verify_rounds:
            # 最后一轮的 compiler 报错作为最终 codesys 错误
            last_round = verify_rounds[-1]
            codesys_errors = last_round.get("compiler", [{"ErrorDesc": "Compilation failed"}])

        entry = {
            "task_id": name,
            "query": description,
            "generated_code": generated_code,
            "antlr_validation": {
                "passed": final_antlr,
                "error": None if final_antlr else "ANTLR syntax check failed",
            },
            "codesys_validation": {
                "task_id": name,
                "compilation_success": final_codesys,
                "errors": codesys_errors,
                "available": codesys_ever_available,
            },
            "task_cost": round(cost, 6),
            "execution_time": per_task_time.get(name, 0.0),
            "prompt_tokens": prompt_tok,
            "completion_tokens": completion_tok,
            "total_tokens": total_tok,
            "true_category": category,
            "method": method,
            "model": model_name,
            "generation_success": generation_success,
            "generation_error": generation_error,
            "is_solved": is_solved,
            "error": None if is_solved else (
                "ANTLR failed" if not final_antlr and final_codesys else
                "CODESYS failed" if not final_codesys and final_antlr else
                "Both failed"
            ),
        }
        results.append(entry)

        # Accumulate category stats
        cat_stats[category]["total"] += 1
        cat_stats[category]["cost"] += cost
        if is_solved:
            cat_stats[category]["solved"] += 1
        if final_antlr:
            cat_stats[category]["antlr_solved"] += 1
        if final_codesys:
            cat_stats[category]["codesys_solved"] += 1
        if final_antlr and final_codesys:
            cat_stats[category]["both_solved"] += 1

        logger.info(f"  {'[OK]' if is_solved else '[FAIL]'} {name:<10}"
                    f" antlr={'PASS' if final_antlr else 'FAIL':<6}"
                    f" cs={'OK' if final_codesys else 'FAIL':<6}"
                    f" rounds={len(task_iterations)}")

    # ── Build summary ──────────────────────────────────────────────────────
    total_tasks = len(results)
    solved = sum(1 for r in results if r["is_solved"])
    antlr_pass = sum(1 for r in results if r["antlr_validation"]["passed"])
    codesys_pass = sum(1 for r in results if r["codesys_validation"]["compilation_success"])
    both_pass = sum(1 for r in results if r["antlr_validation"]["passed"] and r["codesys_validation"]["compilation_success"])
    unsolved = total_tasks - solved
    pass_rate = round(solved / total_tasks * 100, 2) if total_tasks else 0
    total_cost = round(sum(r["task_cost"] for r in results), 6)
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)

    summary = {
        "experiment_summary": {
            "total_tasks": total_tasks,
            "solved_tasks": solved,
            "unsolved_tasks": unsolved,
            "pass_rate": pass_rate,
            "total_execution_time_seconds": round(experiment_total_time, 2),
            "total_cost": total_cost,
            "avg_cost_per_task": round(total_cost / total_tasks, 6) if total_tasks else 0,
            "avg_time_per_task_seconds": round(experiment_total_time / total_tasks, 2) if total_tasks else 0,
            "total_tokens": {
                "prompt": total_prompt,
                "completion": total_completion,
                "total": total_prompt + total_completion,
            },
            "iteration_stats": iter_stats,
            "category_stats": {
                cat: {
                    "total": st["total"],
                    "solved": st["solved"],
                    "antlr_solved": st["antlr_solved"],
                    "codesys_solved": st["codesys_solved"],
                    "both_solved": st["both_solved"],
                    "cost": round(st["cost"], 6),
                    "time": round(st["time"], 4),
                }
                for cat, st in sorted(cat_stats.items())
            },
            "antlr_final_pass": antlr_pass,
            "antlr_final_pass_rate": round(antlr_pass / total_tasks * 100, 2) if total_tasks else 0,
            "codesys_final_pass": codesys_pass,
            "codesys_final_pass_rate": round(codesys_pass / total_tasks * 100, 2) if total_tasks else 0,
            "both_final_pass": both_pass,
            "both_final_pass_rate": round(both_pass / total_tasks * 100, 2) if total_tasks else 0,
            "cost_currency": currency,
        }
    }

    # ── Write output files (直接放在 base_folder 下，llm4plc 风格) ─────────
    # 文件名: autoplc_{model_slug}_{base_folder_ts}.json
    base_name = os.path.basename(base_folder)  # e.g. deepseek-ai-DeepSeek-V3.2_20260507_141545
    model_slug = base_name.rsplit("_", 1)[0] if "_" in base_name else model_name.replace("/", "-").replace(":", "-")

    per_task_file = os.path.join(base_folder, f"autoplc_{base_name}.json")
    summary_file = os.path.join(base_folder, f"autoplc_{base_name}_summary.json")

    with open(per_task_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Per-task results -> {per_task_file}")

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Summary          -> {summary_file}")

    # ── Print summary ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f" AutoPLC Experiment Summary  |  model={model_name}  |  {currency}")
    logger.info("=" * 60)
    logger.info(f"  Total tasks:    {total_tasks}")
    logger.info(f"  ANTLR passed:   {antlr_pass}/{total_tasks} ({round(antlr_pass/total_tasks*100,1) if total_tasks else 0}%)")
    logger.info(f"  CODESYS passed: {codesys_pass}/{total_tasks} ({round(codesys_pass/total_tasks*100,1) if total_tasks else 0}%)")
    logger.info(f"  Both passed:    {both_pass}/{total_tasks} ({round(both_pass/total_tasks*100,1) if total_tasks else 0}%)")
    logger.info(f"  Solved:         {solved}/{total_tasks} ({pass_rate}%)")
    logger.info(f"  Total cost:     {total_cost:.4f} {currency}")
    logger.info(f"  Total tokens:   {total_prompt + total_completion:,}")
    logger.info("=" * 60)

    return base_folder
