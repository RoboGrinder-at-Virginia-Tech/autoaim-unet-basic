import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from armor_unet.data import ArmorDataModule
from armor_unet.lit_module import ArmorUNet


DATA_ROOT = os.environ.get('DATA_ROOT', 'Dataset_Robomaster-1')
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', 'checkpoints')
LOG_DIR = os.environ.get('LOG_DIR', 'logs')


def train_armor_detector(
    data_root=DATA_ROOT,
    batch_size=8,
    max_epochs=1,
    learning_rate=1e-4,
    base_channels=32,
    checkpoint_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
):
    """Complete training pipeline with PyTorch Lightning"""

    pl.seed_everything(42, workers=True)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("="*80)
    print("ARMOR PLATE DETECTION - PyTorch Lightning")
    print("="*80)

    print("\nInitializing data module...")
    datamodule = ArmorDataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=2,
    )

    print("Creating model...")
    model = ArmorUNet(
        learning_rate=learning_rate,
        base_channels=base_channels,
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

    wandb_logger = WandbLogger(
        project='armor-unet',
        save_dir=log_dir,
        log_model=True,
        offline=False,
    )
    wandb_logger.experiment.config.update({
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'base_channels': base_channels,
        'max_epochs': max_epochs,
        'data_root': data_root,
    })
    wandb_logger.watch(model, log='all', log_freq=50)

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
    # Minimal CLI via env vars; users can also edit defaults above
    max_epochs_env = os.environ.get('MAX_EPOCHS')
    if max_epochs_env is not None:
        try:
            me = int(max_epochs_env)
        except ValueError:
            me = 5
        train_armor_detector(max_epochs=me)
    else:
        train_armor_detector()
