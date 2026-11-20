"""Recheck the released Agents4PLC-23 programs with the official verifier."""
from pathlib import Path
import runpy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "evaluation/agents4plc/scripts"))

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[3] / "evaluation/agents4plc/scripts/run_alignment.py"), run_name="__main__")
