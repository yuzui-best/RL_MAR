import subprocess
import os
from pathlib import Path
# -*- coding: utf-8 -*-

TOOLS_DIR = Path(__file__).resolve().parent
bat_path = os.environ.get("AUTOPLC_CODESYS_BAT", str(TOOLS_DIR / "callCodesys.bat"))

test_content = r"""
"""



test_filename =  ""

class State:
    def __init__(self):
        self.filename = ""
        self.content = ""

state = State() 


def run_bat():
    current_path = Path(__file__).resolve().parent.joinpath("codesys_debugger.py")
    # try:
    result = subprocess.run(
        [bat_path],
        shell=True,
        check=True,
        text=True,
        capture_output=True,
        timeout=3600
    )
    return result

def pre_exec(content):
    lines = content.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.upper().startswith('END_FUNCTION_BLOCK'):
            break  
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def post_exec(content):
    error_lines = []
    lines = content.split('\n')
    for line in lines:
        error_pos = line.find(': Error:')
        if error_pos != -1:
            error_content = line[error_pos + len(': Error:'):].lstrip()
            error_lines.append(error_content)

    return error_lines

def codesys_check(filename, content):
    
    temp_dir = Path(os.environ.get("AUTOPLC_TEMP_DIR", str(TOOLS_DIR / "tempfile")))
    temp_dir.mkdir(parents=True, exist_ok=True)

    for item_path in temp_dir.iterdir():
        if item_path.is_file():
            item_path.unlink()
    
    # scl_files = []
    # for entry in os.listdir(temp_dir):
    #     if entry.lower().endswith('.st') and os.path.isfile(os.path.join(temp_dir, entry)):
    #         scl_files.append(entry)
    
    # if not scl_files:
    #     raise FileNotFoundError("tempfile file not found")
    # original_file = os.path.join(temp_dir, scl_files[0])
    file_ext = ".st"
    
    new_filename = f"{filename}{file_ext}"
    new_file_path = temp_dir / new_filename
    with open(new_file_path, 'w', encoding="utf-8") as f:
        f.write(content)

    print(state.filename)
    compile_res = run_bat()
    print(compile_res)
    final_error = post_exec(compile_res.stderr)
    print(final_error)
    return final_error

def get_name_content():
    return test_filename, test_content
# if __name__ == "__main__":
#     codesys_check(test_filename, test_content)


