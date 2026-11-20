"""Public entry point: official verification of retained programs."""
import runpy
from pathlib import Path
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_alignment.py")), run_name="__main__")
