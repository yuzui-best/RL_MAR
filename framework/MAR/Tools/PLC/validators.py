# MAR/Tools/PLC/validators.py

from typing import Dict, List
import re
# 
try:
    from .antlr_compiler import STCompiler
except ImportError:
    from MAR.Tools.PLC.antlr_compiler import STCompiler

# 
_antlr_compiler = STCompiler()

def st_syntax_checker_tool(code: str) -> Dict:
    """
    Validate Structured Text with the ANTLR compiler.

    The adapter strips Markdown fences, decides whether the input is already
    a complete POU or a code fragment, wraps fragments when needed, and returns
    structured validation feedback.
    """
    # Strip Markdown fences.
    if not code:
        return {"passed": False, "error": "Empty input"}
        
    clean_code = code.replace("```st", "").replace("```", "").strip()
    
    if not clean_code:
        return {"passed": False, "error": "Empty code block after cleaning"}

    # Detect whether the code already contains a complete POU declaration.
    has_block_structure = re.search(r'\b(FUNCTION_BLOCK|PROGRAM|FUNCTION)\b', clean_code, re.IGNORECASE)

    if has_block_structure:
        final_code = clean_code
    else:
        # Wrap fragments so the grammar can parse them as a function block.
        final_code = f"""
FUNCTION_BLOCK RL_MAR_Auto_Check
    VAR_TEMP
       // reserved
    END_VAR
    BEGIN
        {clean_code}
    
END_FUNCTION_BLOCK
"""

    result = _antlr_compiler.check_syntax(final_code)
    
    if result['passed']:
        return {
            "passed": True,
            "error": None,
            "feedback": "Syntax check passed successfully."
        }
    else:
        error_msg = "\n".join(result['errors'])
        return {
            "passed": False,
            "error": error_msg,
            "feedback": f"Syntax check failed:\n{error_msg}"
        }

def property_validator_tool(code: str, properties: List) -> Dict:
    """
    Placeholder for future logic verification.
    """
    return {
        "all_passed": True,
        "details": {}
    }
