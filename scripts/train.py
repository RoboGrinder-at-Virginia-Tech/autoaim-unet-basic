import os
import sys
import warnings
# Add project root to sys.path to allow importing armor_unet without installation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Fix for OpenMP runtime conflict (common on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from armor_unet.data import ArmorDataModule
from armor_unet.lit_module import ArmorUNet

# Suppress Triton warning on Windows
warnings.filterwarnings("ignore", message="triton not found")


DATA_ROOT = os.environ.get('DATA_ROOT', 'Dataset_Robomaster-1')
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', 'checkpoints')
LOG_DIR = os.environ.get('LOG_DIR', 'logs')


def train_armor_detector(
    data_root=DATA_ROOT,
    batch_size=8,
    max_epochs=1,
    learning_rate=1e-4,
    weight_decay=1e-5,
    model_name="small", # model size selection
    base_channels=None, # allows for custom base channels
    loss_name="bce",
    checkpoint_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
    project_name="armor-unet",
    run_name=None,  # Add this parameter
):
    """Complete training pipeline with PyTorch Lightning"""

    pl.seed_everything(42, workers=True)
    # added line for torch float32 matmul precision
    torch.set_float32_matmul_precision('high')  # or 'medium' for slower performance

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("="*80)
    print("ARMOR PLATE DETECTION - PyTorch Lightning")
    print("="*80)

    # Windows often crashes with multiprocessing in DataLoaders
    # Default to 1 on Windows, 2 on Linux/Mac
    num_workers = 1 if os.name == 'nt' else 2

    print("\nInitializing data module...")
    datamodule = ArmorDataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print("Creating model...")
    model = ArmorUNet(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        model_name=model_name, # pass model_name to lit_module
        base_channels=base_channels,
        loss_name=loss_name,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='armor-unet-{epoch:02d}-{val_dice:.4f}',
        monitor='val_dice',
        mode='max',
        save_top_k=1,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_dice',
        patience=15,
        mode='max',
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    if run_name:
        print(f"Initializing W&B with custom name: {run_name}")
    
    wandb_logger = WandbLogger(
        project=project_name,
        name=run_name,
        save_dir=log_dir,
        log_model=True,
        offline=False,
        config={
        'model_name': model_name, # model size selection
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'base_channels': base_channels,
        'max_epochs': max_epochs,
        'loss_name': loss_name,
        'data_root': data_root,
        }
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10,
        deterministic=True,
    )

    print("\nStarting training...")
    print("="*80)
    trainer.fit(model, datamodule)
    wandb_logger.watch(model, log='all', log_freq=50)

    print("\n" + "="*80)
    print("Testing best model...")
    trainer.test(model, datamodule, ckpt_path='best')

    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"Best validation Dice: {checkpoint_callback.best_model_score:.4f}")
    print("\nTrack the run in Weights & Biases:")
    print(f"  {wandb_logger.experiment.url}")
    wandb_logger.experiment.finish()

    return model, trainer, datamodule


if __name__ == '__main__':
    import argparse

    # Argument parsing instead of direct parameter setting via environment
    parser = argparse.ArgumentParser(description='Train armor plate detector')
    parser.add_argument('--model', type=str, default='small', help='Model size selection') 
    parser.add_argument('--data-root', type=str, default=DATA_ROOT, help='Path to dataset root')
    parser.add_argument('--project', type=str, default='armor-unet', help='W&B project name')
    parser.add_argument('--run-name', type=str, default=None, help='Custom W&B run name') 
    parser.add_argument('--max-epochs', type=int, default=1, help='Maximum training epochs') 
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate') 
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--loss', type=str, default='bce', help='Loss function (bce, bce_dice, focal)')
    
    args = parser.parse_args()
    
    train_armor_detector(
        model_name=args.model,
        data_root=args.data_root,
        project_name=args.project,
        run_name=args.run_name,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
    )