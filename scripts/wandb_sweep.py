"""
scripts/wandb_sweep.py — Hyperparameter tuning with Weights & Biases Sweeps.
Tunes all model architectures (UNets, MobileNet, TorchVision) on GPU only.
"""
import os
import sys
import argparse
import warnings

# Fix for OpenMP runtime conflict (common on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

warnings.filterwarnings("ignore", message="triton not found")

# Add project root to sys.path to allow importing armor_unet without installation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from armor_unet.data import ArmorDataModule
from armor_unet.lit_module import ArmorUNet

# Suppress Triton warning on Windows
warnings.filterwarnings("ignore", message="triton not found")


def check_cuda():
    """Check if CUDA is available and abort if not."""
    if not torch.cuda.is_available():
        print("=" * 80)
        print("ERROR: CUDA device not found!")
        print("This sweep script requires a GPU with CUDA support.")
        print("=" * 80)
        sys.exit(1)
    
    print("=" * 80)
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print("=" * 80)


def train_sweep(args):
    """
    Single sweep trial that trains with W&B-provided hyperparameters.
    
    Args:
        args: Command-line arguments containing data_root, epochs, etc.
    """
    # Initialize W&B run (W&B will inject the config for this trial)
    run = wandb.init()
    cfg = wandb.config
    
    # Seed for reproducibility
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    
    # Windows often crashes with multiprocessing in DataLoaders
    # Force num_workers=1 on Windows to ensure stability
    num_workers = cfg.get("num_workers", 2)
    if os.name == 'nt':
        num_workers = 1

    # Initialize data module
    datamodule = ArmorDataModule(
        data_root=args.data_root,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
    )
    
    # Determine base_channels (not applicable for mobilenet or torchvision models)
    if cfg.model_name in ["mobilenet", "deeplabv3", "fcn", "lraspp"]:
        base_channels = None
    else:
        base_channels = cfg.base_channels
    
    # Initialize Lightning module
    model = ArmorUNet(
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        base_channels=base_channels,
        model_name=cfg.model_name,
        loss_name=cfg.loss_name,
    )
    
    # Setup W&B logger
    logger = WandbLogger(
        project=args.project,
        save_dir=args.log_dir,
        experiment=run,
        log_model=False,
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_dice",
            mode="max",
            save_top_k=1,
            filename=f"{{epoch}}-{{val_dice:.4f}}-{cfg.model_name}",
            dirpath=os.path.join(args.checkpoint_dir, run.name),
        ),
        EarlyStopping(
            monitor="val_dice",
            patience=10,
            mode="max",
            verbose=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Initialize trainer - GPU ONLY
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
        deterministic=args.deterministic,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    # Fit and test the model
    trainer.fit(model, datamodule=datamodule)
    
    # Test with best checkpoint
    if trainer.checkpoint_callback.best_model_path:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")
    
    # Finish the W&B run
    wandb.finish()


def main():
    """Main function to set up and run W&B Sweep."""
    
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for armor detection using W&B Sweeps"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.getenv("DATA_ROOT", "Dataset_Robomaster-1"),
        help="Path to dataset root directory"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=os.getenv("WANDB_PROJECT", "armor-unet-sweeps"),
        help="W&B project name"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.getenv("LOG_DIR", "logs"),
        help="Directory for Lightning logs"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/sweeps",
        help="Directory for model checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of epochs per trial"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of sweep runs to perform"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic training for reproducibility"
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Existing sweep ID to resume (optional)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="bayes",
        choices=["random", "grid", "bayes"],
        help="Sweep search method"
    )
    
    args = parser.parse_args()
    
    # Check for CUDA availability - ABORT if not found
    check_cuda()
    
    print("\n" + "=" * 80)
    print("WEIGHTS & BIASES HYPERPARAMETER SWEEP")
    print("=" * 80)
    print(f"Dataset: {args.data_root}")
    print(f"Project: {args.project}")
    print(f"Max epochs per trial: {args.epochs}")
    print(f"Number of trials: {args.count}")
    print(f"Search method: {args.method}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("=" * 80 + "\n")
    
    # Define sweep configuration
    sweep_config = {
        "name": "armor-unet-all-models",
        "method": args.method,
        "metric": {
            "name": "val_dice",
            "goal": "maximize"
        },
        "parameters": {
            # Model architecture
            "model_name": {
                "values": ["small", "medium", "large", "mobilenet", "deeplabv3", "fcn", "lraspp"]
            },
            
            # Optimizer hyperparameters
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 5e-3
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-7,
                "max": 1e-3
            },
            
            # Model architecture parameters (for UNet variants)
            # Reduced to avoid OOM on smaller GPUs
            "base_channels": {
                "values": [16, 32]
            },
            
            # Training hyperparameters - reduced batch sizes for GPU memory
            "batch_size": {
                "values": [4, 8]
            },
            
            # Data loading
            "num_workers": {
                "values": [2, 4]
            },
            
            # Loss function
            "loss_name": {
                "values": ["bce", "bce_dice", "focal"]
            },
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 5,
            "eta": 2,
        }
    }
    
    # Create or use existing sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Resuming existing sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print(f"Created new sweep: {sweep_id}")
    
    print(f"View sweep at: https://wandb.ai/{wandb.api.default_entity}/{args.project}/sweeps/{sweep_id}")
    print("\n" + "=" * 80)
    print(f"Starting W&B agent with {args.count} trials...")
    print("=" * 80 + "\n")
    
    # Run the sweep agent
    wandb.agent(
        sweep_id,
        function=lambda: train_sweep(args),
        count=args.count,
        project=args.project
    )
    
    print("\n" + "=" * 80)
    print("SWEEP COMPLETE!")
    print("=" * 80)
    print(f"View results at: https://wandb.ai/{wandb.api.default_entity}/{args.project}/sweeps/{sweep_id}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
