import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .models import UNet


def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


class ArmorUNet(pl.LightningModule):
    """PyTorch Lightning module for armor plate detection"""

    def __init__(self, learning_rate=1e-4, weight_decay=1e-5, base_channels=32,
                model_name='small', loss_name: str = 'bce', loss_params: dict | None = None):
        super().__init__()
        self.save_hyperparameters()

        # added import statement for models.py
        from .models import get_model
        self.model = get_model(
            model_name=model_name,
            in_channels=3,
            out_channels=1,
            base_channels=base_channels
        )

        self.loss_name = loss_name
        self.loss_params = loss_params or {}
        self._init_criterion()

        self.example_input_array = torch.randn(1, 3, 640, 640)

    def _init_criterion(self):
        name = (self.loss_name or 'bce').lower()
        if name == 'bce':
            bce = nn.BCEWithLogitsLoss()
            self.criterion = lambda logits, target: bce(logits, target)
        elif name == 'bce_dice':
            dice_lambda = float(self.loss_params.get('dice_lambda', 1.0))
            bce = nn.BCEWithLogitsLoss()
            self.criterion = lambda logits, target, w=dice_lambda: bce(logits, target) + w * self.soft_dice_loss(logits, target)
        elif name == 'focal':
            alpha = float(self.loss_params.get('alpha', 0.25))
            gamma = float(self.loss_params.get('gamma', 2.0))
            self.criterion = lambda logits, target, a=alpha, g=gamma: self.focal_loss_with_logits(logits, target, a, g)
        elif name == 'tversky':
            alpha = float(self.loss_params.get('alpha', 0.3))
            beta = float(self.loss_params.get('beta', 0.7))
            gamma = float(self.loss_params.get('gamma', 1.33))
            self.criterion = lambda logits, target, a=alpha, b=beta, g=gamma: self.tversky_loss(logits, target, a, b, g)
        else:
            bce = nn.BCEWithLogitsLoss()
            self.criterion = lambda logits, target: bce(logits, target)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        loss = self.criterion(outputs, masks)
        dice = dice_coefficient(outputs, masks)

        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_dice', dice, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        loss = self.criterion(outputs, masks)
        dice = dice_coefficient(outputs, masks)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, 'val_dice': dice}

    def test_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        loss = self.criterion(outputs, masks)
        dice = dice_coefficient(outputs, masks)

        self.log('test_loss', loss)
        self.log('test_dice', dice)

        return {'test_loss': loss, 'test_dice': dice}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_dice',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    @staticmethod
    def _flatten_probs_and_targets(logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        return probs, targets

    def soft_dice_loss(self, logits, targets, smooth: float = 1e-6):
        probs, targets = self._flatten_probs_and_targets(logits, targets)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def focal_loss_with_logits(self, logits, targets, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-6):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t).clamp(min=eps) ** gamma) * bce
        return loss.mean()

    def tversky_loss(self, logits, targets, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1e-6):
        probs, targets = self._flatten_probs_and_targets(logits, targets)
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        loss = (1.0 - tversky) ** gamma
        return loss.mean()
