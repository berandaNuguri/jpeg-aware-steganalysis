"""Select a source module; pass all remaining arguments unchanged."""
import os
from pathlib import Path
import argparse
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODULES = {'png': 'jpeg_aware_steganalysis.utils.data_processing.manifests.png', 'jpeg': 'jpeg_aware_steganalysis.utils.data_processing.manifests.jpeg', 'path_modify': 'jpeg_aware_steganalysis.utils.data_processing.manifests.path_modify', 'jpeg_imagename_only': 'jpeg_aware_steganalysis.utils.data_processing.manifests.jpeg_imagename_only', 'png_imagename_only': 'jpeg_aware_steganalysis.utils.data_processing.manifests.png_imagename_only', 'bossbase_bows2': 'jpeg_aware_steganalysis.utils.data_processing.manifests.bossbase_bows2'}

if __name__ == "__main__":
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=sorted(MODULES))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    selected = parser.parse_args()
    sys.argv = [MODULES[selected.tool]] + selected.arguments
    runpy.run_module(MODULES[selected.tool], run_name="__main__")
