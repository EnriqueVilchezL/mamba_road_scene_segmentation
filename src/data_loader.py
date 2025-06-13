import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pytorch_lightning import LightningDataModule
from configuration import COLOR_MAP


def rgb_to_class(mask):
    """Converts a 3-channel RGB mask to a single-channel class index mask."""
    mask_np = np.array(mask)
    class_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
    for rgb, idx in COLOR_MAP.items():
        matches = np.all(mask_np == rgb, axis=-1)
        class_mask[matches] = idx
    return class_mask


def class_to_rgb(class_mask):
    """Converts a single-channel class index mask to a 3-channel RGB mask."""
    rgb_mask = np.zeros((class_mask.shape[0], class_mask.shape[1], 3), dtype=np.uint8)
    for rgb, idx in COLOR_MAP.items():
        rgb_mask[class_mask == idx] = rgb
    return rgb_mask


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, f) for f in self.ids]
        self.masks_fps = [
            os.path.join(masks_dir, Path(f).stem + "_train_color.png") for f in self.ids
        ]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.images_fps[i]).convert("RGB")
        mask = Image.open(self.masks_fps[i]).convert("RGB")

        if self.transform:
            img, mask = self.transform(img, mask)
        else:
            img = transforms.ToTensor()(img)
            mask = torch.from_numpy(rgb_to_class(mask)).long()

        return img, mask


class TestSegmentationDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, f) for f in self.ids]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.images_fps[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img


class SegmentationTransform:
    def __init__(self, image_size=(256, 256)):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        )
        self.mask_resize = transforms.Resize(image_size, interpolation=Image.NEAREST)

    def __call__(self, img, mask):
        img = self.image_transform(img)
        mask = self.mask_resize(mask)
        mask = torch.from_numpy(rgb_to_class(mask)).long()
        return img, mask


class SegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size=8,
        data_path="data",
        transform=None,
        num_classes=12,
        num_workers=4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = Path(data_path)
        self.transform = transform
        self.num_classes = num_classes
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = SegmentationDataset(
            images_dir=self.data_path / "images" / "train",
            masks_dir=self.data_path / "masks" / "train",
            transform=self.transform,
        )
        self.val_dataset = SegmentationDataset(
            images_dir=self.data_path / "images" / "val",
            masks_dir=self.data_path / "masks" / "val",
            transform=self.transform,
        )
        self.test_dataset = TestSegmentationDataset(
            images_dir=self.data_path / "images" / "test",
            transform=self.transform.image_transform if self.transform else None,
        )

    def train_dataloader(self):
        return self.val_dataloader()
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            # num_workers=self.num_workers,
            # persistent_workers=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()
        # return DataLoader(
        #     self.test_dataset,
        #     batch_size=self.batch_size,
        #     # num_workers=self.num_workers,
        #     # persistent_workers=True,
        # )
