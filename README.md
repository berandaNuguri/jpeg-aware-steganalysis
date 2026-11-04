# JPEG-Aware Steganalysis

Research code for **[JPEG-aware steganalysis: explicit frequency modeling with dequantized DCT for real-world JPEG images](https://doi.org/10.1038/s41598-026-59285-1)**.

Dong-Su Kim, Jin-Cheol An, Youngjoon Yoo, Tae-Woo Oh, and Haneol Jang.  
*Scientific Reports*, 2026.

## Overview

JPEG steganalysis must account for more than image content: resolution, quantization tables and chroma subsampling can all change the statistical evidence available to a detector. This study examines these compression-related distribution shifts through **RWSteg**, a real-world evaluation protocol grounded in web imagery and empirical JPEG-header statistics.

The framework combines metadata-guided compression alignment, block-aligned region selection and **RWNet**, a detector that explicitly models JPEG frequency information.

## Main components

- **RWSteg protocol:** evaluates heterogeneous resolutions, quantization tables and chroma subsampling, including mismatched and targeted training/evaluation conditions.
- **JPEG-aware representation:** reconstructs dequantized DCT coefficients using image-specific quantization tables and handles luminance/chrominance components.
- **RWNet:** combines JPEG-structured preprocessing with spatial pyramid pooling to aggregate frequency-domain features.
- **ROI evaluation:** selects original-resolution, block-aligned local evidence without globally resizing the image. ROI selection is a framework-level step, separate from RWNet's internal pooling.

## Usage

Task scripts resolve relative paths from the repository root. Local datasets belong under `src/data/` and checkpoints under `checkpoints/`; both are excluded from Git. Set `RESEARCH_DATA_ROOT` / `RESEARCH_MODEL_ROOT` or pass explicit CLI paths to use external resources. These locations do not include datasets or weights automatically.

Prepare an environment with the dependencies declared in pyproject.toml. Supply images, CSV manifests and checkpoints separately, and select the matching model configuration.

```bash
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/prepare_data.py --help
```

Additional model-specific entry points support SF3Net and Two-stream experiments. Datasets and trained checkpoints are external resources. Refer to the [published article](https://www.nature.com/articles/s41598-026-59285-1) for protocol definitions, experimental conditions and results.
