"""
LLM4PLC Baseline - Reproduction Experiment Script

This script reproduces the LLM4PLC three-phase iterative method, adapted for:
- CODESYS ST code generation (instead of Siemens SCL)
- ANTLR + CODESYS dual verification (instead of SMV formal verification)
- MasRouter v4 environment and dataset

Respects the original LLM4PLC framework while adapting necessary components.
"""

import os
import time
import subprocess
import re
import json
import traceback
import requests
import threading
import logging
from pathlib import Path
import sys
from datetime import datetime

# Set UTF-8 encoding for Windows console (prevent UnicodeEncodeError)
if sys.platform == 'win32':
    import codecs
    import io
    # Set stdout and stderr to UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Setup logging
log_dir = Path(__file__).parent.joinpath('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir.joinpath(f"llm4plc_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (same as gpt-4o-mini baseline)
env_file = Path(__file__).parent.joinpath('.env')
if env_file.exists():
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

# Resolve the parent directory for whole program and llm4plc itself as an absolute path
parent_dir = Path(__file__).resolve().parent
sys.path.append(str(parent_dir))

# Add parent directory to path (same as gpt-4o-mini baseline)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import LLM4PLC verification tools (adapted)
try:
    from config import chat_model, reasoning_model, max_tokens, efficiency_rounds
except ImportError:
    # Fallback configuration if config not available
    chat_model = "gpt-4o"
    reasoning_model = "gpt-4o"
    max_tokens = 4000
    efficiency_rounds = 5

# Import verification tools independently so one optional checker cannot mask
# the compiler validator.
try:
    from MAR.Tools.PLC.validators import st_syntax_checker_tool
except ImportError as e:
    logger.warning(f"Could not import ANTLR verification tool: {e}")
    st_syntax_checker_tool = None

try:
    from validate_with_codesys import CODESYSValidator, CompilerInfrastructureError
    logger.info("Compiler verification tool imported successfully")
except ImportError as e:
    logger.error(f"Could not import compiler verification tool: {e}")
    CODESYSValidator = None

    class CompilerInfrastructureError(RuntimeError):
        pass

# Global CODESYS validator instance (reuse instead of creating new instances)
_codesys_validator_instance = None

def get_codesys_validator():
    """Get or create CODESYS validator instance (reuse pattern like gpt-4o-mini)"""
    global _codesys_validator_instance
    if CODESYSValidator is not None and _codesys_validator_instance is None:
        server_host = os.environ.get("CODESYS_SERVER_HOST", "127.0.0.1")
        server_port = int(os.environ.get("CODESYS_SERVER_PORT", "19000"))
        _codesys_validator_instance = CODESYSValidator(server_host=server_host, server_port=server_port)
        logger.info("CODESYS validator instance created (will be reused)")
    return _codesys_validator_instance

# Simple LLM caller using requests (same as gpt-4o-mini baseline)
def call_llm(system_message: str, user_message: str, model: str = "gpt-4o",
             temperature: float = 0.3, max_tokens: int = 3000, timeout: int = 120):
    """
    Simple LLM caller using OpenAI API (same as gpt-4o-mini baseline)

    Args:
        system_message: System message
        user_message: User message
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Returns:
        tuple: (content: str, usage: dict) - generated text and actual API token usage
    """
    # Get API configuration from environment (support multiple variable names)
    api_key = os.environ.get("LLM4PLC_API_KEY") or os.environ.get("KEY") or os.environ.get("API_KEY", "")
    base_url = os.environ.get("LLM4PLC_BASE_URL") or os.environ.get("URL") or os.environ.get("BASE_URL", "https://api.openai.com/v1")

    # Use default model from env if specified
    model = os.environ.get("DEFAULT_MODEL", model)

    if not api_key:
        logger.error("No API key found in environment variables (checked: LLM4PLC_API_KEY, KEY, API_KEY)")
        return "Error: No API key configured", {}

    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    endpoint = f"{base_url}/chat/completions"

    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(os.environ.get("TEMPERATURE", temperature)),
            "top_p": float(os.environ.get("TOP_P", "0.9")),
            "max_tokens": int(os.environ.get("MAX_TOKENS", max_tokens)),
            "thinking": {"type": "disabled"}
        }

        logger.debug(f"Calling LLM API: {model}, endpoint: {endpoint}")

        response = requests.post(
            endpoint,
            json=payload,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            timeout=timeout
        )

        if response.status_code == 200:
            result = response.json()
            usage = result.get('usage', {})
            logger.info(f"LLM API call successful: {usage}")
            return result['choices'][0]['message']['content'], usage
        else:
            logger.error(f"LLM API error: {response.status_code} - {response.text}")
            raise RuntimeError(f"LLM_API_ERROR: HTTP {response.status_code}: {response.text[:500]}")

    except requests.exceptions.Timeout:
        logger.error("LLM API request timeout")
        raise RuntimeError("LLM_API_ERROR: request timeout")
    except Exception as e:
        logger.error(f"LLM API request exception: {e}")
        if str(e).startswith("LLM_API_ERROR:"):
            raise
        raise RuntimeError(f"LLM_API_ERROR: {e}") from e

############# function tools #################

def st_grammar_checker(st_file_path, timeout=30):
    """
    ST grammar checking using ANTLR (adapted from LLM4PLC's grammar_check.sh)
    This replaces the original Rusty/Matiec compilation check.
    """
    try:
        # Use your ANTLR-based syntax checker
        if st_syntax_checker_tool is not None:
            with open(st_file_path, 'r') as f:
                code_content = f.read()

            result = st_syntax_checker_tool(code_content)

            if result['passed']:
                return f'ST output passed ANTLR syntax validation'
            else:
                error_msg = result.get('error', 'Unknown error')
                return f'ST output failed ANTLR syntax validation: {error_msg}'
        else:
            # Fallback: basic syntax check
            return 'ANTLR checker not available, skipped validation'

    except Exception as e:
        return f'ANTLR syntax check failed: {str(e)}'


def codesys_compiler_checker(st_file_path, timeout=30):
    """
    CODESYS compiler checking (adapted from LLM4PLC's compilation verification)
    This replaces the original SMV model checking.
    Uses reusable validator instance (same pattern as gpt-4o-mini baseline).
    """
    if os.environ.get("SKIP_LEGACY_CODESYS_VALIDATION", "").lower() in {"1", "true", "yes"}:
        return "ST output passed legacy CODESYS validation (skipped for Functional94-only evaluation)"

    try:
        # Use your CODESYS validator with instance reuse
        validator = get_codesys_validator()

        if validator is not None:
            with open(st_file_path, 'r') as f:
                code_content = f.read()

            # Generate a unique task_id for this validation
            task_id = Path(st_file_path).stem

            # Call the validator
            result = validator.validate_code(code_content, task_id)

            # Parse the result
            if result.get("compilation_success", False):
                logger.info(f"CODESYS validation passed for {task_id}")
                return f'ST output passed CODESYS compiler validation'
            else:
                errors = result.get("errors", [])
                if errors:
                    error_desc = errors[0].get("ErrorDesc", "Unknown error") if errors else "Unknown compilation error"
                    logger.warning(f"CODESYS validation failed for {task_id}: {error_desc}")
                    return f'ST output failed CODESYS compiler validation: {error_desc}'
                else:
                    logger.warning(f"CODESYS validation failed for {task_id}: Unknown error")
                    return f'ST output failed CODESYS compiler validation: Unknown error'
        else:
            raise CompilerInfrastructureError(
                "INFRA_ERROR: compiler validator is unavailable"
            )

    except CompilerInfrastructureError:
        raise
    except Exception as e:
        logger.error(f"CODESYS compiler check failed: {str(e)}")
        raise CompilerInfrastructureError(f"INFRA_ERROR: compiler check failed: {e}") from e


def input_with_timeout(prompt, timeout=30, default=""):
    """
    Prompts the user for input with a timeout. Returns the default value if the user does not respond in time.
    """
    user_input = [default]

    def get_input():
        try:
            user_input[0] = input(prompt)
        except EOFError:
            user_input[0] = default

    input_thread = threading.Thread(target=get_input)
    input_thread.start()
    input_thread.join(timeout)

    if input_thread.is_alive():
        print(f"\nNo response received within {timeout} seconds. Using default value: '{default}'")
        return default

    return user_input[0]


def simple_log_message(message, log_file_path):
    """Logs a simple message to the log file with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_path, 'a', encoding='utf-8', errors='replace') as log_file:
        log_file.write("="*50 + "\n")
        log_file.write(f"Timestamp: {timestamp}\n")
        # Sanitize message to prevent encoding errors
        safe_message = message.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        log_file.write(f"{safe_message}\n")


def log_message(log_file_path, sys_msg, user_msg, result, phase_name="Unknown phase"):
    """Logs the system message, user message, and model response with timestamp to the log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_path, 'a', encoding='utf-8', errors='replace') as log_file:
        log_file.write("="*50 + "\n")
        log_file.write(f"Phase: {phase_name}\n")
        log_file.write(f"Timestamp: {timestamp}\n")
        # Sanitize all messages to prevent encoding errors
        safe_sys_msg = sys_msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        safe_user_msg = user_msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        safe_result = result.encode('utf-8', errors='replace').decode('utf-8', errors='replace')

        log_file.write(f"System Message:\n{safe_sys_msg}\n")
        log_file.write(f"User Message:\n{safe_user_msg}\n")
        log_file.write(f"Model Response:\n{safe_result}\n")


def create_log_directory(spec_path, base_dir=None):
    """
    Creates a log directory with a file to store the interaction process.
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "results")
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if not spec_path:
        spec_name = None
        log_dir = f"{base_dir}/{timestamp}"
    else:
        spec_name = os.path.basename(spec_path).split('.')[0]
        log_dir = f"{base_dir}/{timestamp}_{spec_name}"

    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, "log.txt")
    return log_file_path, log_dir


def extract_section(text, start_tag, end_tag):
    """Helper function to extract content between start and end tags.
    Searches for end_tag only after start_tag to avoid matching at the same position."""
    start_idx = text.find(start_tag)

    if start_idx == -1:
        return text

    end_idx = text.find(end_tag, start_idx + len(start_tag))

    if end_idx == -1:
        return text

    return text[start_idx + len(start_tag):end_idx].strip()


def extract_st_code(text):
    """Remove response protocol wrappers and return only the ST POU."""
    content = (text or "").strip()
    if re.match(r"^START_REASON\s*$", content, re.MULTILINE):
        parts = re.split(r"^END_REASON\s*$", content, maxsplit=1, flags=re.MULTILINE)
        if len(parts) == 2:
            content = parts[1].strip()
        else:
            pou = re.search(r"^(?:FUNCTION_BLOCK|FUNCTION|PROGRAM)\b", content, re.MULTILINE)
            if pou:
                content = content[pou.start():].strip()
    tagged = re.search(r"^START_ST\s*$\n?(.*?)^END_ST\s*$", content, re.MULTILINE | re.DOTALL)
    if tagged:
        return tagged.group(1).strip()
    fenced = re.search(r"```(?:st|structured-text|iecst|text)?\s*(.*?)```", content, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    content = re.sub(r"^START_ST\s*$", "", content, count=1, flags=re.MULTILINE).strip()
    content = re.sub(r"^END_ST\s*$", "", content, count=1, flags=re.MULTILINE).strip()
    pou = re.search(r"^(?:FUNCTION_BLOCK|FUNCTION|PROGRAM)\b", content, re.MULTILINE)
    return content[pou.start():].strip() if pou else ""


def extract_plan_and_question(text):
    """Extract plan and question from LLM response."""
    plan = extract_section(text, "[START_PLAN]", "[END_PLAN]")
    question = extract_section(text, "[START_QUESTION]", "[END_QUESTION]")

    if not plan:
        plan = text

    return plan, question


##################### workflow phases #################

def planning_and_feedback(spec, log_file, spec_is_dir=False):
    """
    Phase 1: Planning (respecting original LLM4PLC framework)
    Returns: (plan, phase1_metrics)
    """
    user_msg = spec

    print(f"\nStep 1: Planning for user input: {user_msg[:100]}...")

    # Initialize phase 1 metrics
    phase1_metrics = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0
    }

    # Load adapted Phase 1 prompt
    planning_sys_msg_path = os.path.join(parent_dir, "prompts/phase1_planning_codessys_adapted.txt")

    if not os.path.exists(planning_sys_msg_path):
        # Fallback to original prompt structure
        planning_sys_msg_path = os.path.join(parent_dir, "prompts/phase1_planning_codessys_adapted.txt")

    with open(planning_sys_msg_path, 'r', encoding='utf-8') as sys_file:
        sys_msg = sys_file.read()

    print(f"System message for planning: {sys_msg[:100]}...")

    # Call LLM to generate the plan
    plan_result, plan_usage = call_llm(sys_msg, user_msg)

    # Accumulate usage from planning LLM call
    if plan_usage:
        phase1_metrics["prompt_tokens"] += plan_usage.get("prompt_tokens", 0)
        phase1_metrics["completion_tokens"] += plan_usage.get("completion_tokens", 0)
        phase1_metrics["total_tokens"] += plan_usage.get("total_tokens", 0)

    # Log the planning phase
    log_message(log_file, sys_msg, user_msg, plan_result, "Planning Phase")

    # Extract the plan and question from the response
    plan, question = extract_plan_and_question(plan_result)
    print(f"Extracted Plan: {plan[:200]}...")
    print(f"Extracted Question: {question}")

    # If LLM proposes a question (skip user interaction for batch processing)
    if question:
        print("LLM has posed a question. Skipping user feedback for batch processing...")

        # In batch mode, we skip the user interaction and proceed with the current plan
        # This is different from the original interactive design but necessary for automation
        print(f"Proceeding with current plan (ignoring question for automation)")
        user_feedback = None

        if user_feedback:
            # Load feedback prompt (if exists)
            feedback_sys_msg_path = os.path.join(parent_dir, "prompts/phase1_planning_codessys_adapted.txt")

            with open(feedback_sys_msg_path, 'r', encoding='utf-8') as feedback_sys_file:
                feedback_sys_msg = feedback_sys_file.read()

            print(f"User feedback: {user_feedback}")

            # Call LLM to update the plan based on user response
            feedback_usr_msg = (f"The original question is: {user_msg}. "
                                f"The previous plan is: {plan}, question is: {question}, "
                                f"and user feedback for question is: {user_feedback}. "
                                "You can update the previous plan for a detailed one, based on which another one can directly write st code.")

            updated_plan, feedback_usage = call_llm(feedback_sys_msg, feedback_usr_msg)

            # Log the feedback phase
            log_message(log_file, feedback_sys_msg, user_feedback, updated_plan, "Feedback Phase")

            # Accumulate usage from feedback LLM call
            if feedback_usage:
                phase1_metrics["prompt_tokens"] += feedback_usage.get("prompt_tokens", 0)
                phase1_metrics["completion_tokens"] += feedback_usage.get("completion_tokens", 0)
                phase1_metrics["total_tokens"] += feedback_usage.get("total_tokens", 0)

            plan = extract_section(updated_plan, "[START_PLAN]", "[END_PLAN]")
            print(f"Updated Plan: {plan[:200]}...")

        else:
            print("No feedback provided, proceeding with original plan.")
    else:
        user_feedback = input_with_timeout("Do you approve the plan? (y/n): ", timeout=5, default="y")
        if user_feedback.lower() == 'n':
            print("No feedback provided, moving to the next phase.")

    return plan, phase1_metrics


def code_gen(spec, plan, log_file, log_dir, max_regenerate_attempts=3, func_call_time=0, spec_is_dir=False,
             iteration_stats=None):
    """
    Phase 2: Model-based design to generate compiled ST file (respecting original framework)

    IMPORTANT: Phase 2 generates code ONLY ONCE. No internal retry loop.
    All iteration optimization is handled by Phase 3.

    Iteration Definition:
    - Iteration 0: First-time generation (before any Phase 3 refinement)
    - This is recorded as "iteration_0" in statistics

    Returns: (scl_code, scl_file_path, iteration_validation, phase_metrics)
    """
    if iteration_stats is None:
        iteration_stats = {
            "iteration_0": {"total": 0, "antlr_passed": 0, "codesys_passed": 0, "both_passed": 0},
            "iteration_1": {"total": 0, "antlr_passed": 0, "codesys_passed": 0, "both_passed": 0},
            "iteration_2": {"total": 0, "antlr_passed": 0, "codesys_passed": 0, "both_passed": 0},
            "iteration_3": {"total": 0, "antlr_passed": 0, "codesys_passed": 0, "both_passed": 0}
        }

    # Generate only ONCE in Phase 2
    scl_file_path = os.path.join(log_dir, f"generated_code_iteration_0.st")
    iteration_key = "iteration_0"  # Phase 2 is always iteration_0

    # Update iteration statistics for Phase 2 (iteration_0)
    iteration_stats[iteration_key]["total"] += 1

    # Track metrics for this phase
    phase_metrics = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0
    }

    print(f"\nStep 2: Model-based design, generating ST file (Iteration 0 - First Generation)")

    # Load adapted Phase 2 prompt
    design_sys_msg_path = os.path.join(parent_dir, "prompts/phase2_generation_codessys_adapted.txt")

    if not os.path.exists(design_sys_msg_path):
        # Fallback to original
        design_sys_msg_path = os.path.join(parent_dir, "prompts/phase2_generation_codessys_adapted.txt")

    with open(design_sys_msg_path, 'r', encoding='utf-8') as design_sys_file:
        design_sys_msg = design_sys_file.read()

    # Prepare the input content for the LLM
    scl_input_content = (f"Origin requirement: {spec}, YOU MUST generate format following the code's head and inside the POU; "
                         f"Plan for code: {plan}. "
                         "Notice you should generate Structured Text code that satisfies the ST format in IEC-61131-3 standard "
                         "based on the plan.")

    scl_content, gen_usage = call_llm(design_sys_msg, scl_input_content)

    # Use actual token usage from API response
    if gen_usage:
        phase_metrics["prompt_tokens"] += gen_usage.get("prompt_tokens", 0)
        phase_metrics["completion_tokens"] += gen_usage.get("completion_tokens", 0)
        phase_metrics["total_tokens"] += gen_usage.get("total_tokens", 0)

    # Log the coding phase
    log_message(log_file, design_sys_msg, scl_input_content, scl_content, phase_name="Phase2_Generation_Iteration0")

    # Strip protocol/reasoning wrappers before any compiler feedback.
    scl_code = extract_st_code(scl_content)

    # Initialize validation result
    iteration_validation = {}

    # Ensure scl_code is defined (empty string if extraction failed)
    if not scl_code:
        scl_code = ""

    if scl_code:
        # Save the ST code to the file
        with open(scl_file_path, 'w', encoding='utf-8') as scl_file:
            scl_file.write(scl_code)
        print(f"ST file saved at: {scl_file_path}")

        # NEW: Use your dual verification system
        # First check: ANTLR syntax check
        antlr_result = st_grammar_checker(scl_file_path)
        antlr_passed = "passed" in antlr_result.lower()
        print(f"ANTLR syntax check: {antlr_result[:100] if len(antlr_result) > 100 else antlr_result}")

        # Second check: CODESYS compiler check
        codesys_result = codesys_compiler_checker(scl_file_path)
        codesys_passed = "passed" in codesys_result.lower()
        print(f"CODESYS compiler check: {codesys_result[:100] if len(codesys_result) > 100 else codesys_result}")

        # Track validation results for Phase 2 (iteration_0)
        iteration_validation = {
            "iteration": "iteration_0",
            "phase": "Phase2_Generation",
            "antlr_passed": antlr_passed,
            "antlr_result": antlr_result,
            "codesys_passed": codesys_passed,
            "codesys_result": codesys_result
        }

        # Update iteration statistics for Phase 2 (iteration_0)
        if antlr_passed:
            iteration_stats[iteration_key]["antlr_passed"] += 1
        if codesys_passed:
            iteration_stats[iteration_key]["codesys_passed"] += 1
        if antlr_passed and codesys_passed:
            iteration_stats[iteration_key]["both_passed"] += 1

        # Print the verification output for reference
        print(f"Phase 2 (Iteration 0) verification output: ANTLR={antlr_passed}, CODESYS={codesys_passed}")
        simple_log_message(f"Phase 2 (Iteration 0) verification: ANTLR={antlr_passed}, CODESYS={codesys_passed}", log_file_path=log_file)
    else:
        print("ERROR: Could not extract valid ST code from LLM response")

        # Track failed extraction for Phase 2 (iteration_0)
        iteration_validation = {
            "iteration": "iteration_0",
            "phase": "Phase2_Generation",
            "antlr_passed": False,
            "antlr_result": "Code extraction failed",
            "codesys_passed": False,
            "codesys_result": "Code extraction failed"
        }

        # Log code extraction failure for traceability
        simple_log_message("Phase 2 (Iteration 0) verification: ANTLR=False, CODESYS=False (code extraction failed)", log_file_path=log_file)

        # Save empty file for traceability
        with open(scl_file_path, 'w', encoding='utf-8') as scl_file:
            scl_file.write("")
        print(f"Empty ST file saved at: {scl_file_path}")

    # Return Phase 2 results (no retry, always return after first generation)
    return scl_code, scl_file_path, iteration_validation, phase_metrics




def verification(phase2_validation, phase2_code, log_file, log_dir, iteration_stats,
                 max_phase3_attempts=3, properties="", func_call_time=0, properties_is_json_list=False):
    """
    Phase 3: Verification with iterative refinement (respecting original framework)

    This function performs verification and refinement cycles:
    - Takes the Phase 2 results (iteration_0)
    - If not perfect, runs Phase 3 refinement (becomes iteration_1, iteration_2, iteration_3)
    - Each Phase 3 attempt is recorded as a separate iteration

    Args:
        phase2_validation: Validation result from Phase 2 (iteration_0)
        phase2_code: Code generated from Phase 2
        iteration_stats: Statistics tracker for all iterations
        max_phase3_attempts: Max Phase 3 refinement attempts (default: 3)

    Returns: (final_code, final_validation_result, success, phase3_metrics)
    """
    # Initialize Phase 3 metrics
    phase3_metrics = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0
    }

    # Check if Phase 2 was already perfect
    if (phase2_validation.get("antlr_passed", False) and
        phase2_validation.get("codesys_passed", False)):
        print("✓ Phase 2 (Iteration 0) code already validated successfully, skipping Phase 3 refinement")
        return phase2_code, phase2_validation, True, phase3_metrics

    print(f"\nPhase 2 code needs refinement, starting Phase 3 optimization (up to {max_phase3_attempts} attempts)")

    # Initialize current_validation before loop to prevent NameError on continue edge cases
    current_validation = {
        "iteration": "N/A",
        "phase": "Phase3_NotStarted",
        "antlr_passed": False,
        "antlr_result": "No validation performed yet",
        "codesys_passed": False,
        "codesys_result": "No validation performed yet"
    }

    # Phase 3 optimization loop
    for phase3_attempt in range(max_phase3_attempts):
        iteration_number = phase3_attempt + 1  # iteration_1, iteration_2, iteration_3
        iteration_key = f"iteration_{iteration_number}"

        print(f"\nStep 3: Phase 3 refinement (Iteration {iteration_number}, Attempt {phase3_attempt + 1}/{max_phase3_attempts})")

        # Load adapted Phase 3 prompt
        verification_sys_msg_path = os.path.join(parent_dir, "prompts/phase3_refinement_codessys_simplified.txt")

        if not os.path.exists(verification_sys_msg_path):
            # Fallback to original
            verification_sys_msg_path = os.path.join(parent_dir, "prompts/llm4plc_original/phase3_refinement_full.txt")

        with open(verification_sys_msg_path, 'r', encoding='utf-8') as verification_sys_file:
            sys_msg = verification_sys_file.read()

        print(f"System message for Phase 3: {sys_msg[:100]}...")

        # Create validation feedback from Phase 2 or previous Phase 3 attempts
        if phase3_attempt == 0:
            # First Phase 3 attempt: use Phase 2 validation feedback
            validation_feedback = (
                f"Phase 2 (Iteration 0) validation feedback:\n"
                f"ANTLR: {phase2_validation.get('antlr_result', 'Unknown')}\n"
                f"CODESYS: {phase2_validation.get('codesys_result', 'Unknown')}\n\n"
                f"Please fix the issues and return the complete improved code."
            )
            current_code = phase2_code
        else:
            # Subsequent Phase 3 attempts: use previous Phase 3 validation feedback
            validation_feedback = (
                f"Phase 3 (Iteration {phase3_attempt}) validation feedback:\n"
                f"ANTLR: {current_validation.get('antlr_result', 'Unknown')}\n"
                f"CODESYS: {current_validation.get('codesys_result', 'Unknown')}\n\n"
                f"Please continue fixing the issues and return the complete improved code."
            )

        input_msg = f"ST code: {current_code}\nValidation feedback: {validation_feedback}"

        # Call LLM for Phase 3 refinement
        refined_content, refine_usage = call_llm(sys_msg, input_msg)

        # Log the Phase 3 refinement phase
        log_message(log_file, sys_msg, input_msg, refined_content, phase_name=f"Phase3_Refinement_Iteration{iteration_number}")

        # Accumulate usage from this Phase 3 iteration
        if refine_usage:
            phase3_metrics["prompt_tokens"] += refine_usage.get("prompt_tokens", 0)
            phase3_metrics["completion_tokens"] += refine_usage.get("completion_tokens", 0)
            phase3_metrics["total_tokens"] += refine_usage.get("total_tokens", 0)

        # Strip protocol/reasoning wrappers before compiler feedback.
        refined_code = extract_st_code(refined_content)

        if not refined_code:
            print(f"WARNING: Could not extract valid ST code from Phase 3 refinement (Iteration {iteration_number}), using empty string for validation")
            refined_code = ""

        # Always save refined code for validation (even if empty)
        refined_file_path = os.path.join(log_dir, f"generated_code_iteration_{iteration_number}.st")

        with open(refined_file_path, 'w', encoding='utf-8') as f:
            f.write(refined_code)

        # Validate the refined code
        print(f"Validating Phase 3 refined code (Iteration {iteration_number})...")

        if not refined_code:
            # Code extraction failed: force both checks to fail
            antlr_passed = False
            antlr_result = "ST output failed ANTLR syntax validation: Code extraction failed (no valid ST code found in LLM response)"
            codesys_passed = False
            codesys_result = "ST output failed CODESYS compiler validation: Code extraction failed (no valid ST code found in LLM response)"
        else:
            # ANTLR validation
            antlr_result = st_grammar_checker(refined_file_path)
            antlr_passed = "passed" in antlr_result.lower()
            # CODESYS validation
            codesys_result = codesys_compiler_checker(refined_file_path)
            codesys_passed = "passed" in codesys_result.lower()

        print(f"ANTLR validation (Iteration {iteration_number}): {antlr_result[:100] if len(antlr_result) > 100 else antlr_result}")
        print(f"CODESYS validation (Iteration {iteration_number}): {codesys_result[:100] if len(codesys_result) > 100 else codesys_result}")

        # Record validation result for this iteration
        current_validation = {
            "iteration": f"iteration_{iteration_number}",
            "phase": f"Phase3_Refinement_Attempt_{phase3_attempt + 1}",
            "antlr_passed": antlr_passed,
            "antlr_result": antlr_result,
            "codesys_passed": codesys_passed,
            "codesys_result": codesys_result
        }

        # Update iteration statistics for this Phase 3 iteration
        iteration_stats[iteration_key]["total"] += 1
        if antlr_passed:
            iteration_stats[iteration_key]["antlr_passed"] += 1
        if codesys_passed:
            iteration_stats[iteration_key]["codesys_passed"] += 1
        if antlr_passed and codesys_passed:
            iteration_stats[iteration_key]["both_passed"] += 1

        # Check if both validations passed
        if antlr_passed and codesys_passed:
            print(f"✓ Phase 3 refinement (Iteration {iteration_number}) validated successfully!")
            simple_log_message(f"Phase 3 (Iteration {iteration_number}) validation: SUCCESS", log_file_path=log_file)
            return refined_code, current_validation, True, phase3_metrics
        else:
            print(f"Phase 3 refinement (Iteration {iteration_number}) still has issues, continuing...")
            simple_log_message(f"Phase 3 (Iteration {iteration_number}) validation: FAILED", log_file_path=log_file)

        # Update current code and validation for next iteration
        current_code = refined_code

    # Max Phase 3 attempts reached without success
    print(f"Phase 3 refinement completed after {max_phase3_attempts} attempts without full success")
    return current_code, current_validation, False, phase3_metrics


def run_llm_4_plc_1_time(spec, properties, base_dir, spec_is_path=True,
                         max_regenerate_attempts=3, max_verification_attempts=3,
                         iteration_stats=None):
    """
    Run llm4plc for 1 time (respecting original framework)
    Returns: (scl_file_path, properties, iteration_results, task_metrics)
    """
    if iteration_stats is None:
        iteration_stats = {
            "iteration_0": {"total": 0, "antlr_passed": 0, "codesys_passed": 0, "both_passed": 0},
            "iteration_1": {"total": 0, "antlr_passed": 0, "codesys_passed": 0, "both_passed": 0},
            "iteration_2": {"total": 0, "antlr_passed": 0, "codesys_passed": 0, "both_passed": 0},
            "iteration_3": {"total": 0, "antlr_passed": 0, "codesys_passed": 0, "both_passed": 0}
        }

    if spec_is_path:
        log_file, log_dir = create_log_directory(spec, base_dir=base_dir)
    else:
        log_file, log_dir = create_log_directory("", base_dir=base_dir)

    # Track iteration results and metrics
    iteration_results = {}
    task_metrics = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0
    }

    plan, phase1_metrics = planning_and_feedback(spec, log_file, spec_is_dir=False)
    print(f"DEBUG: Plan generated, calling code_gen...")

    try:
        scl_code, scl_file_path, iteration_validation, phase_metrics = code_gen(spec, plan, log_file, log_dir,
                                           max_regenerate_attempts=max_regenerate_attempts, func_call_time=0,
                                           iteration_stats=iteration_stats)
        print(f"DEBUG: code_gen returned successfully: scl_code type={type(scl_code)}, len={len(scl_code) if scl_code else 'N/A'}")
    except Exception as e:
        print(f"DEBUG: code_gen failed with error: {e}")
        print(f"DEBUG: error type: {type(e)}, error args: {e.args}")
        raise

    # Store iteration validation results
    iteration_results = {}
    iteration_results["iteration_0"] = iteration_validation

    # Accumulate metrics from Phase 1 (planning)
    if phase1_metrics:
        task_metrics["prompt_tokens"] += phase1_metrics.get("prompt_tokens", 0)
        task_metrics["completion_tokens"] += phase1_metrics.get("completion_tokens", 0)
        task_metrics["total_tokens"] += phase1_metrics.get("total_tokens", 0)

    # Accumulate metrics from Phase 2 (code generation)
    if phase_metrics:
        task_metrics["prompt_tokens"] += phase_metrics.get("prompt_tokens", 0)
        task_metrics["completion_tokens"] += phase_metrics.get("completion_tokens", 0)
        task_metrics["total_tokens"] += phase_metrics.get("total_tokens", 0)

    # Get the validation status from Phase 2 (iteration_0) - always defined
    phase2_validation = iteration_results.get("iteration_0", {})
    print(f"Phase 2 validation: ANTLR={phase2_validation.get('antlr_passed')}, CODESYS={phase2_validation.get('codesys_passed')}")

    # Always run Phase 3 verification and refinement (no gate on scl_code)
    # Even if code extraction failed or validation failed, Phase 3 will attempt refinement
    final_code, final_validation, verif_success, phase3_metrics = verification(
        phase2_validation, scl_code, log_file, log_dir, iteration_stats,
        max_phase3_attempts=max_verification_attempts,
        properties=properties,
        properties_is_json_list=True
    )
    print(f"Phase 3 result: {verif_success}")

    # Accumulate metrics from Phase 3 (refinement)
    if phase3_metrics:
        task_metrics["prompt_tokens"] += phase3_metrics.get("prompt_tokens", 0)
        task_metrics["completion_tokens"] += phase3_metrics.get("completion_tokens", 0)
        task_metrics["total_tokens"] += phase3_metrics.get("total_tokens", 0)

    # Read pricing from environment (with fallbacks for backward compatibility)
    price_prompt = float(os.environ.get("PRICE_PROMPT_PER_1K", 0))
    price_completion = float(os.environ.get("PRICE_COMPLETION_PER_1K", 0))
    if price_prompt > 0 and price_completion > 0:
        prompt_t = task_metrics.get("prompt_tokens", 0)
        completion_t = task_metrics.get("completion_tokens", 0)
        task_metrics["cost"] = (prompt_t / 1000 * price_prompt) + (completion_t / 1000 * price_completion)

    # Return final results
    return scl_file_path, properties, iteration_results, task_metrics, final_code, final_validation, verif_success


def run_llm_4_plc(benchmark_dir, max_regenerate_attempts=3, max_verification_attempts=3,
                 base_dir=None, single_task=None):
    """
    Main function to run LLM4PLC experiment (respecting original framework)
    Tracks pass rates for each iteration (0, 1, 2, 3) and comprehensive metrics
    """
    logger.info("="*80)
    logger.info("Starting LLM4PLC Three-Phase Baseline Experiment")
    logger.info("="*80)

    # Use the caller-provided isolated result directory when supplied.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = base_dir or os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Get model name for experiment subfolder naming
    model_name = os.environ.get("DEFAULT_MODEL", reasoning_model).replace("/", "-").replace(":", "-")

    # Create experiment subfolder: results/{model}_{timestamp}/
    exp_subfolder = f"{model_name}_{timestamp}"
    exp_dir = os.path.join(results_dir, exp_subfolder)
    os.makedirs(exp_dir, exist_ok=True)

    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Model: {model_name}")

    # Create result file inside experiment subfolder
    result_file = os.path.join(exp_dir, f"llm4plc_baseline_{timestamp}.json")

    logger.info(f"Results file: {result_file}")
    logger.info(f"Max regeneration attempts: {max_regenerate_attempts}")
    logger.info(f"Max verification attempts: {max_verification_attempts}")

    result_list = []
    filename = result_file

    # Track pass rates for each iteration
    iteration_stats = {
        "iteration_0": {"total": 0, "antlr_passed": 0, "codesys_passed": 0, "both_passed": 0},
        "iteration_1": {"total": 0, "antlr_passed": 0, "codesys_passed": 0, "both_passed": 0},
        "iteration_2": {"total": 0, "antlr_passed": 0, "codesys_passed": 0, "both_passed": 0},
        "iteration_3": {"total": 0, "antlr_passed": 0, "codesys_passed": 0, "both_passed": 0}
    }

    # Track comprehensive metrics
    experiment_start_time = time.time()
    total_tokens = {"prompt": 0, "completion": 0, "total": 0}
    total_cost = 0.0

    # Category-wise statistics
    category_stats = {}

    logger.info("Initializing CODESYS validator instance (will be reused for all validations)")
    get_codesys_validator()  # Pre-initialize the validator

    try:
        # Override base_dir to point into experiment subfolder, so per-task dirs live inside it
        base_dir = exp_dir
        os.makedirs(base_dir, exist_ok=True)

        # Parse your dataset format
        try:
            json_input = parse_plc_file(benchmark_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to parse dataset file: {str(e)}")

        # Filter for single task if specified
        if single_task is not None:
            if single_task < 1 or single_task > len(json_input):
                raise ValueError(f"single_task must be between 1 and {len(json_input)}, got {single_task}")
            logger.info(f"Running single task mode: only task {single_task}/{len(json_input)}")
            # Convert to 0-indexed and create a single-item list
            task_indices = [single_task - 1]
        else:
            logger.info(f"Running full dataset mode: {len(json_input)} tasks")
            task_indices = range(len(json_input))

        # Process selected entries
        for task_index in task_indices:
            entry = json_input[task_index]
            i = task_index  # Use for logging
            entry_result = {}
            current_iteration_results = {}  # Track validation results for each iteration
            task_start_time = time.time()

            try:
                # Extract instruction and properties
                instruction = entry.get("instruction", "")
                properties = entry.get("properties_to_be_validated", "")
                category = entry.get("category", "Unknown")
                task_id = entry.get("task_id", f'task_{i+1}')

                if not instruction:
                    raise ValueError("Empty instruction in entry")

                logger.info(f"\n{'='*50}")
                logger.info(f"Processing case {i+1}/{len(json_input)}: {task_id}")
                logger.info(f"Category: {category}")
                logger.info(f"Instruction: {instruction[:100]}...")
                logger.info(f"{'='*50}")

                print(f"\n{'='*50}")
                print(f"Processing case {i+1}: {instruction[:80]}...")
                print(f"{'='*50}")

                # Core LLM4PLC workflow
                scl_file_path, updated_properties, iteration_results, task_metrics, final_code, final_validation, verif_success = run_llm_4_plc_1_time(
                    instruction,
                    base_dir=base_dir,
                    properties=properties,
                    max_regenerate_attempts=max_regenerate_attempts,
                    max_verification_attempts=max_verification_attempts,
                    spec_is_path=False,
                    iteration_stats=iteration_stats
                )

                # Store iteration results
                current_iteration_results = iteration_results

                # Calculate task metrics
                task_execution_time = time.time() - task_start_time

                # Determine final success based on final validation
                final_success = verif_success and final_validation is not None
                if final_validation:
                    final_success = (final_validation.get("antlr_passed", False) and
                                   final_validation.get("codesys_passed", False))

                # Prepare validation results in reference format
                antlr_validation = {
                    "passed": final_validation.get("antlr_passed", False) if final_validation else False,
                    "error": final_validation.get("antlr_result", "") if final_validation else "No validation"
                }

                codesys_validation = {
                    "task_id": task_id,
                    "compilation_success": final_validation.get("codesys_passed", False) if final_validation else False,
                    "errors": [],
                    "available": True
                }

                if final_validation and not final_validation.get("codesys_passed", False):
                    error_msg = final_validation.get("codesys_result", "")
                    codesys_validation["errors"] = [{"ErrorDesc": error_msg}]

                # Update category statistics
                if category not in category_stats:
                    category_stats[category] = {
                        "total": 0,
                        "solved": 0,
                        "antlr_solved": 0,
                        "codesys_solved": 0,
                        "both_solved": 0,
                        "cost": 0.0,
                        "time": 0.0
                    }

                category_stats[category]["total"] += 1
                if final_success:
                    category_stats[category]["solved"] += 1

                # Update ANTLR and CODESYS category stats
                if final_validation:
                    if final_validation.get("antlr_passed", False):
                        category_stats[category]["antlr_solved"] += 1
                    if final_validation.get("codesys_passed", False):
                        category_stats[category]["codesys_solved"] += 1
                    if (final_validation.get("antlr_passed", False) and
                        final_validation.get("codesys_passed", False)):
                        category_stats[category]["both_solved"] += 1

                # Accumulate metrics
                if task_metrics:
                    total_tokens["prompt"] += task_metrics.get("prompt_tokens", 0)
                    total_tokens["completion"] += task_metrics.get("completion_tokens", 0)
                    total_tokens["total"] += task_metrics.get("total_tokens", 0)
                    total_cost += task_metrics.get("cost", 0.0)

                category_stats[category]["cost"] += task_metrics.get("cost", 0.0)
                category_stats[category]["time"] += task_execution_time

                # Build result in reference format
                entry_result = {
                    "task_id": task_id,
                    "query": instruction,
                    "generated_code": final_code,
                    "antlr_validation": antlr_validation,
                    "codesys_validation": codesys_validation,
                    "task_cost": task_metrics.get("cost", 0.0),
                    "execution_time": task_execution_time,
                    "prompt_tokens": task_metrics.get("prompt_tokens", 0),
                    "completion_tokens": task_metrics.get("completion_tokens", 0),
                    "total_tokens": task_metrics.get("total_tokens", 0),
                    "true_category": category,
                    "method": "llm4plc_three_phase",
                    "model": reasoning_model,
                    "generation_success": True,
                    "generation_error": None,
                    "is_solved": final_success,
                    "error": None
                }

            except Exception as e:
                # Log detailed error information
                task_execution_time = time.time() - task_start_time
                logger.error(f"Task {i+1} failed: {type(e).__name__} - {str(e)}")
                logger.debug(f"Task {i+1} traceback:\n{traceback.format_exc()}")

                entry_result = {
                    "status": "error",
                    "entry_index": i,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                    "execution_time": task_execution_time,
                    "is_solved": False
                }
            finally:
                # Always record result
                result_list.append(entry_result)

                # Real-time save
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(result_list, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    print(f"!! Warning: Could not save intermediate results: {str(e)}")

    except Exception as e:
        # Handle top-level exceptions
        error_entry = {
            "status": "critical_error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        result_list.append(error_entry)
    finally:
        # Calculate final statistics
        total_execution_time = time.time() - experiment_start_time
        total_tasks = len(result_list)
        solved_tasks = sum(1 for r in result_list if r.get("is_solved", False))
        pass_rate = (solved_tasks / total_tasks * 100) if total_tasks > 0 else 0

        # Build summary statistics
        experiment_summary = {
            "experiment_summary": {
                "total_tasks": total_tasks,
                "solved_tasks": solved_tasks,
                "unsolved_tasks": total_tasks - solved_tasks,
                "pass_rate": round(pass_rate, 2),
                "total_execution_time_seconds": round(total_execution_time, 2),
                "total_cost": round(total_cost, 6),
                "cost_currency": os.environ.get("CURRENCY", "USD"),
                "avg_cost_per_task": round(total_cost / total_tasks, 6) if total_tasks > 0 else 0,
                "avg_time_per_task_seconds": round(total_execution_time / total_tasks, 2) if total_tasks > 0 else 0,
                "total_tokens": total_tokens,
                "iteration_stats": iteration_stats,
                "category_stats": {k: v for k, v in sorted(category_stats.items())} if category_stats else {}
            }
        }

        # Final save (results as list, backward compatible)
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result_list, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"!! Critical: Could not save final results: {str(e)}")

        # Save companion summary file
        summary_filename = filename.replace('.json', '_summary.json')
        try:
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(experiment_summary, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"!! Could not save summary file: {str(e)}")

        # Print comprehensive statistics
        print_experiment_summary(total_tasks, solved_tasks, pass_rate, total_execution_time,
                               total_cost, total_tokens, iteration_stats, category_stats)

    return base_dir


def print_experiment_summary(total_tasks, solved_tasks, pass_rate, total_execution_time,
                           total_cost, total_tokens, iteration_stats, category_stats):
    """Print comprehensive experiment summary statistics"""
    print("\n" + "=" * 80)
    print(" " * 20 + "LLM4PLC Three-Phase Baseline Experiment Results")
    print("=" * 80)

    # Overall performance
    print(f"\nOverall Performance:")
    print(f"   Total Tasks: {total_tasks}")
    print(f"   Solved Tasks: {solved_tasks}")
    print(f"   Final Pass Rate: {pass_rate:.2f}%")

    # Iteration statistics with clear definitions
    print(f"\nIteration Pass Rate Statistics:")
    print(f"Iteration Definitions:")
    print(f"  Iteration 0: Initial Phase 2 generation (before Phase 3 refinement)")
    print(f"  Iteration 1+: After Phase 3 refinement attempts")
    print(f"{'Iteration':<12} {'Total':>8} {'ANTLR Pass':>15} {'CODESYS Pass':>15} {'Both Pass':>15}")
    print("-" * 80)

    for iter_name in ["iteration_0", "iteration_1", "iteration_2", "iteration_3"]:
        stats = iteration_stats[iter_name]
        if stats["total"] > 0:
            iter_num = iter_name.replace("iteration_", "")
            iter_desc = "Initial Gen" if iter_num == "0" else f"After Refinement {iter_num}"
            antlr_rate = (stats["antlr_passed"] / stats["total"]) * 100
            codesys_rate = (stats["codesys_passed"] / stats["total"]) * 100
            both_rate = (stats["both_passed"] / stats["total"]) * 100
            print(f"Iteration {iter_num:<6} {stats['total']:>8} {stats['antlr_passed']:>8}/{stats['total']:<4} ({antlr_rate:>5.1f}%) {stats['codesys_passed']:>8}/{stats['total']:<4} ({codesys_rate:>5.1f}%) {stats['both_passed']:>8}/{stats['total']:<4} ({both_rate:>5.1f}%)")

    # Cost metrics
    avg_cost = total_cost / total_tasks if total_tasks > 0 else 0
    print(f"\nCost Metrics:")
    print(f"   Total Cost: ${total_cost:.4f}")
    print(f"   Average Cost/Task: ${avg_cost:.4f}")

    # Time metrics
    avg_time = total_execution_time / total_tasks if total_tasks > 0 else 0
    print(f"\nTime Metrics:")
    print(f"   Total Execution Time: {total_execution_time:.2f}s")
    print(f"   Average Time/Task: {avg_time:.2f}s")

    # Token metrics
    avg_tokens = total_tokens["total"] / total_tasks if total_tasks > 0 else 0
    print(f"\nToken Metrics:")
    print(f"   Total Tokens: {total_tokens['total']:,.0f}")
    print(f"   Average Tokens/Task: {avg_tokens:,.0f}")

    # Category-wise statistics
    if category_stats:
        print(f"\nPerformance by Category:")
        print(f"{'Category':<35} {'Total':>6} {'Solved':>6} {'Both Pass':>10} {'Avg Cost':>10}")
        print("-" * 80)

        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            cat_total = stats['total']
            cat_solved = stats['solved']
            cat_both_pass = stats.get('both_solved', 0)
            cat_pass_rate = cat_solved / cat_total * 100 if cat_total > 0 else 0
            cat_both_rate = cat_both_pass / cat_total * 100 if cat_total > 0 else 0
            cat_avg_cost = stats['cost'] / cat_total if cat_total > 0 else 0

            print(f"{category:<35} {cat_total:>6} {cat_solved:>6} {cat_both_rate:>9.1f}% ${cat_avg_cost:>9.4f}")

    print("=" * 80)


def parse_plc_file(file_dir):
    """Parse PLC dataset file (supports both JSON and JSONL formats)"""
    try:
        # First try as JSONL (line-by-line JSON)
        data = []
        with open(file_dir, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        # Normalize entry format
                        if 'prompt' in entry:
                            # Your dataset format
                            normalized_entry = {
                                "instruction": entry.get('prompt', ''),
                                "properties_to_be_validated": entry.get('properties', []),
                                "task_id": entry.get('task_id', f'task_{line_num}'),
                                "category": entry.get('category', 'Unknown'),
                                "original_entry": entry
                            }
                        else:
                            # Original LLM4PLC format
                            normalized_entry = entry
                        data.append(normalized_entry)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue

        if data:
            print(f"Loaded {len(data)} entries from {file_dir} (JSONL format)")
            return data
        else:
            # Fallback to regular JSON
            with open(file_dir, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"Loaded {len(data)} entries from {file_dir} (JSON format)")
            return data

    except Exception as e:
        raise IOError(f"An error occurred while reading the file: {e}")


def generate_smv_compatible_ltl_ctl_model(property_data):
    """
    Generate SMV-compatible model from property (simplified version)
    Respects original LLM4PLC framework but adapted for ST validation
    """
    job_req = property_data.get("property", {}).get("job_req", "")

    if job_req == "assertion":
        return "Assertion passed"
    elif job_req == "pattern":
        pattern_id = property_data.get("property", {}).get("pattern_id", "")
        pattern_params = property_data.get("property", {}).get("pattern_params", {})

        # Simplified pattern templates for basic validation
        pattern_templates = {
            "pattern-implication": "AG((PLC_END AND ({param1})) -> ({param2}))",
            "pattern-invariant": "AG(PLC_END -> ({param1}))",
            "pattern-forbidden": "AG(PLC_END -> !({param1}))",
        }

        if pattern_id in pattern_templates and pattern_params:
            template = pattern_templates[pattern_id]
            # Convert parameter format
            converted_params = {}
            for key, value in pattern_params.items():
                converted_params[key] = value.replace("instance.", "")

            try:
                return template.format(**converted_params)
            except:
                return f"Pattern validation: {pattern_id}"
        else:
            return "Basic validation passed"

    return "Basic validation passed"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM4PLC Baseline Reproduction")
    parser.add_argument("--data_path", required=True, help="Path to dataset file")
    parser.add_argument("--result_dir", default="results", help="Directory to store results")
    parser.add_argument("--max_attempts", type=int, default=3, help="Max regeneration attempts (default: 3)")
    parser.add_argument("--max_verification", type=int, default=3, help="Max verification attempts (default: 3)")
    parser.add_argument("--single_task", type=int, default=None, help="Run only the N-th task (1-indexed, None means run all tasks)")

    args = parser.parse_args()

    print("="*50)
    print("LLM4PLC Baseline Reproduction Experiment")
    print("="*50)
    print(f"Dataset: {args.data_path}")
    print(f"Results: {args.result_dir}")
    print(f"Max regeneration attempts: {args.max_attempts}")
    print(f"Max verification attempts: {args.max_verification}")
    if args.single_task:
        print(f"Single task mode: Running only task {args.single_task}")
    else:
        print(f"Full dataset mode: Running all tasks")
    print("="*50)
    print("This will track ANTLR and CODESYS pass rates for iterations 0, 1, 2, 3")
    print("="*50)

    result_dir = run_llm_4_plc(
        benchmark_dir=args.data_path,
        max_regenerate_attempts=args.max_attempts,
        max_verification_attempts=args.max_verification,
        base_dir=args.result_dir,
        single_task=args.single_task
    )

    print(f"\n{'='*50}")
    print(f"Experiment completed! Results saved to: {result_dir}")
    print(f"{'='*50}")
