# Known Issue
When running a training loop with a model size of "small" and a "batch-size" of 4 or larger, their is an explicit cuDNN error that is caused. This raises the "illegal CUDA operation" flag. I am further investigating a fix for this issue as it may be graphics memory dependent.

# Armor U-Net (autoaim-unet-basic)

Small U-Net segmentation training using PyTorch Lightning. This repository provides a compact training pipeline for armor-plate segmentation. The core code is intentionally lightweight and does not depend on Ray or other heavy HPO frameworks.

## Quick overview

- Train and evaluate a small U-Net using `pytorch-lightning`.
- Dataset: COCO-style folders (`train/`, `valid/`, `test/`) with `_annotations.coco.json`.
- Optional experiment logging with Weights & Biases (W&B).
- Reproducible environment via `environment.yml` (conda) and `requirements.txt` (pip).

## Prerequisites

- Git
- Conda (or mamba) for the recommended environment flow, or Python 3.10+ for pip/venv installs.

## Install (recommended: conda/mamba)

Create the environment using the included `environment.yml`:

```powershell
mamba env create -f environment.yml
# or with conda:
conda env create -f environment.yml
```

Activate it:

```powershell
mamba activate armor-unet
# or:
conda activate armor-unet
```

Install the package in editable mode:
```
pip install -e .
```
This installs armor_unet as a package, allowing import it from anywhere in the environment (e.g., from armor_unet.models import UNet). The -e flag changes to take effect immediately without reinstalling.

To update an existing environment from the file:

```powershell
conda env update -n armor-unet -f environment.yml --prune
```

## Alternative: pip + venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
pip install -r requirements.txt
```

## Dataset layout

Place your dataset in a root directory (default: `Dataset_Robomaster-1`) and follow COCO-style layout:

```
Dataset_Robomaster-1/
  train/
    _annotations.coco.json
    img_0001.jpg
    ...
  valid/
    _annotations.coco.json
  test/
    _annotations.coco.json
```

The training code reads `DATA_ROOT` from the environment by default. You can override it with an environment variable or pass args if implemented in scripts.

## Run training

Example (PowerShell):

```powershell
$env:DATA_ROOT = 'C:\path\to\Dataset_Robomaster-1'
$env:CHECKPOINT_DIR = 'checkpoints'
$env:LOG_DIR = 'logs'
python train.py
```

## Logging (Weights & Biases)

This project integrates with W&B via Lightning's `WandbLogger`.

1. Login once with `wandb login` or set `WANDB_API_KEY`.
2. To run offline: set `WANDB_MODE=offline` and later run `wandb sync` to upload.

Run artifacts and metrics are saved under `logs/` during training and local W&B run directories (these are ignored by git).

## Notes about Roboflow

`roboflow` is included in the environment and may be used for dataset downloads/management in custom scripts. If you rely on a specific `roboflow` API version, pin it in `environment.yml` or `requirements.txt` for reproducibility.

## Project structure

```
armor_unet/         # core package: data, model, LightningModule
  __init__.py
  data.py            # Dataset and LightningDataModule
  lit_module.py      # LightningModule and metrics
  models.py          # U-Net components
train.py             # Training entrypoint
requirements.txt
environment.yml
scripts/             # optional helpers (e.g., tuning templates)
notebooks/           # example notebooks
logs/                # runtime logs and checkpoints
```

## Development for future improvements

- Keep heavy external tooling (Ray, custom HPO) isolated in `scripts/` so the main training code remains lightweight.
- If you want a Ray Tune template, add it as a separate script that calls the existing training functions.

## Troubleshooting

- If CUDA is not detected, ensure your PyTorch build matches your system CUDA runtime or use the conda `pytorch-cuda` package in `environment.yml`.
- If package installation fails via conda, the `pip:` block in `environment.yml` provides a fallback.
