import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl


class ArmorPlateDataset(Dataset):
    """Dataset for armor plate segmentation from COCO format"""

    def __init__(self, root_dir, split='train', augment=False):
        self.root_dir = root_dir
        self.split = split
        self.img_dir = os.path.join(root_dir, split)

        # Load COCO annotations
        ann_path = os.path.join(self.img_dir, '_annotations.coco.json')
        with open(ann_path, 'r') as f:
            self.coco_data = json.load(f)

        # Create image_id to annotations mapping
        self.img_id_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

        self.images = self.coco_data['images']

        # Albumentations transforms
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_height = img_info['height']
        img_width = img_info['width']

        # Load image
        img_path = os.path.join(self.img_dir, img_filename)
        image = np.array(Image.open(img_path).convert('RGB'))

        # Create binary mask from bounding boxes
        mask = np.zeros((img_height, img_width), dtype=np.float32)
        if img_id in self.img_id_to_anns:
            for ann in self.img_id_to_anns[img_id]:
                bbox = ann['bbox']
                x, y, w, h = map(int, bbox)
                mask[y:y+h, x:x+w] = 1.0

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].unsqueeze(0)

        return image, mask


class ArmorDataModule(pl.LightningDataModule):
    """Lightning DataModule for armor detection"""

    def __init__(self, data_root, batch_size=8, num_workers=2):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ArmorPlateDataset(self.data_root, split='train', augment=True)
            self.val_dataset = ArmorPlateDataset(self.data_root, split='valid', augment=False)

        if stage == 'test' or stage is None:
            self.test_dataset = ArmorPlateDataset(self.data_root, split='test', augment=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
