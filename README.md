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
python scripts/train.py
```

## Hyperparameter Tuning with W&B Sweeps

This project includes a W&B Sweeps script for automated hyperparameter optimization across all four model architectures (small, medium, large, mobilenet).

### Prerequisites for Sweeps
1. Install and login to W&B:
   ```powershell
   wandb login
   ```
2. Ensure you have a CUDA-capable GPU (the sweep script will abort if CUDA is not available)

### Running a Sweep

**Basic usage** (20 trials with Bayesian optimization):
```powershell
python scripts/wandb_sweep.py --count 20 --epochs 20
```

**Custom configuration**:
```powershell
python scripts/wandb_sweep.py `
  --data-root Dataset_Robomaster-1 `
  --project armor-unet-sweeps `
  --count 30 `
  --epochs 25 `
  --method bayes `
  --deterministic
```

**Resume an existing sweep**:
```powershell
python scripts/wandb_sweep.py --sweep-id <sweep-id> --count 10
```

### Available Arguments
- `--data-root`: Path to dataset (default: `Dataset_Robomaster-1`)
- `--project`: W&B project name (default: `armor-unet-sweeps`)
- `--epochs`: Max epochs per trial (default: 20)
- `--count`: Number of trials to run (default: 20)
- `--method`: Search method - `random`, `grid`, or `bayes` (default: `bayes`)
- `--sweep-id`: Resume existing sweep by ID
- `--deterministic`: Enable deterministic training
- `--checkpoint-dir`: Directory for model checkpoints (default: `checkpoints/sweeps`)
- `--log-dir`: Directory for Lightning logs (default: `logs`)

### What Gets Tuned
The sweep optimizes the following hyperparameters:
- **Model architecture**: small, medium, large, mobilenet
- **Learning rate**: 1e-5 to 5e-3 (log-uniform)
- **Weight decay**: 1e-7 to 1e-3 (log-uniform)
- **Base channels**: 16, 32, 64 (for UNet variants)
- **Batch size**: 4, 8, 12, 16
- **Number of workers**: 2, 4
- **Loss function**: bce, bce_dice, focal

The sweep uses Hyperband early termination to stop poorly performing trials and maximize `val_dice` score.

### Running Multiple Agents (Parallel Sweeps)
To run multiple trials in parallel across different GPUs or machines:

1. Create the sweep (run once):
   ```powershell
   python scripts/wandb_sweep.py --count 1
   # Note the sweep ID from output
   ```

2. On each machine/GPU, run an agent:
   ```powershell
   python scripts/wandb_sweep.py --sweep-id <sweep-id> --count 5
   ```

Each agent will pull trials from the sweep queue and execute them independently.

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
