"""Run the namespaced implementation from the project checkout."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    os.chdir(ROOT)
    runpy.run_module('jpeg_aware_steganalysis.engine.evaluate_sf3net', run_name="__main__")
