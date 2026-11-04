"""Select a source module; pass all remaining arguments unchanged."""
import os
from pathlib import Path
import argparse
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODULES = {'feat_projection': 'jpeg_aware_steganalysis.analysis.feat_projection', 'visualization_2': 'jpeg_aware_steganalysis.analysis.visualization_2', 'jpeg_qtable_inventory': 'jpeg_aware_steganalysis.analysis.jpeg_qtable_inventory', 'visualization': 'jpeg_aware_steganalysis.analysis.visualization', 'qtable_unit_perturbation_analysis': 'jpeg_aware_steganalysis.analysis.qtable_unit_perturbation_analysis', 'roi_extraction_time_benchmark': 'jpeg_aware_steganalysis.analysis.statistics.roi_extraction_time_benchmark', 'net_analysis': 'jpeg_aware_steganalysis.analysis.statistics.net_analysis', 'roi_visaulization': 'jpeg_aware_steganalysis.analysis.statistics.roi_visaulization', 'size_inform': 'jpeg_aware_steganalysis.analysis.statistics.size_inform'}

if __name__ == "__main__":
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=sorted(MODULES))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    selected = parser.parse_args()
    sys.argv = [MODULES[selected.tool]] + selected.arguments
    runpy.run_module(MODULES[selected.tool], run_name="__main__")
