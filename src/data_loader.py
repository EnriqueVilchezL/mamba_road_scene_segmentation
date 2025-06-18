import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from pytorch_lightning import LightningDataModule
from configuration import COLOR_MAP, CLASS_WEIGHTS
import random
from tqdm import tqdm

def rgb_to_class(mask):
    """Converts a 3-channel RGB mask to a class index mask using nearest RGB match."""
    mask_np = np.array(mask)
    h, w, _ = mask_np.shape
    class_mask = np.zeros((h, w), dtype=np.int64)

    color_keys = np.array(list(COLOR_MAP.keys()))  # Shape: [N_classes, 3]
    color_vals = np.array(list(COLOR_MAP.values()))  # Shape: [N_classes]

    # Reshape mask to [H*W, 3]
    flat_mask = mask_np.reshape(-1, 3)

    # Compute L2 distances between each pixel and known class RGBs
    dists = np.linalg.norm(flat_mask[:, None, :] - color_keys[None, :, :], axis=2)  # Shape: [H*W, N_classes]

    # Find nearest color
    nearest_color_indices = np.argmin(dists, axis=1)
    class_mask = color_vals[nearest_color_indices].reshape(h, w)

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

class AddGaussianNoise:
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        np_img = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(self.mean, self.std, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 1)
        return Image.fromarray((np_img * 255).astype(np.uint8))

class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, target):
        angle = random.uniform(-self.degrees, self.degrees) if isinstance(self.degrees, (int, float)) else random.uniform(*self.degrees)
        image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR)
        target = F.rotate(target, angle, interpolation=F.InterpolationMode.NEAREST)
        return image, target

class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.size)
        image = F.crop(image, i, j, h, w)
        target = F.crop(target, i, j, h, w)
        return image, target

class SegmentationTrainTransform:
    def __init__(self, image_size=(256, 256)):
        self.image_resize = transforms.Compose(
            [
                transforms.Resize((256,256)),
            ]
        )
        self.image_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                AddGaussianNoise(0.0, 0.03),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ]
        )
        self.mask_resize = transforms.Resize((256,256), interpolation=Image.NEAREST)
        self.shared_transforms = DualCompose(
            [
                RandomHorizontalFlip(flip_prob=0.8),
                RandomCrop(image_size),
                RandomRotation(degrees=0.30)
            ]
        )


    def __call__(self, img, mask):
        img = self.image_resize(img)
        mask = self.mask_resize(mask)

        # Apply shared augmentations
        img, mask = self.shared_transforms(img, mask)

        img = self.image_transform(img)
        mask = torch.from_numpy(rgb_to_class(mask)).long()
        return img, mask

class SegmentationTestTransform:
    def __init__(self, image_size=(256, 256)):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
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
        train_transform=None,
        test_transform=None,
        num_classes=12,
        num_workers=4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = Path(data_path)
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.num_classes = num_classes
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = SegmentationDataset(
            images_dir=self.data_path / "images" / "train",
            masks_dir=self.data_path / "masks" / "train",
            transform=self.train_transform,
        )
        self.val_dataset = SegmentationDataset(
            images_dir=self.data_path / "images" / "val",
            masks_dir=self.data_path / "masks" / "val",
            transform=self.test_transform,
        )
        self.test_dataset = TestSegmentationDataset(
            images_dir=self.data_path / "images" / "test",
            transform=self.test_transform.image_transform if self.test_transform else None,
        )

    def get_class_weights(self):
        class_counts = torch.zeros(self.num_classes)

        for _, mask in tqdm(self.train_dataset, desc="Computing class frequencies"):
            # Ensure mask is a tensor of shape (H, W) with class indices
            if isinstance(mask, torch.Tensor):
                mask = mask.long()
            else:
                mask = torch.from_numpy(mask).long()

            # Count frequencies of each class in this mask
            for cls in range(self.num_classes):
                class_counts[cls] += (mask == cls).sum()

        # Avoid division by zero
        class_counts[class_counts == 0] = 1

        # Inverse frequency (higher weight = rarer class)
        weights = 1.0 / class_counts
        weights = weights / weights.sum()  # Normalize

        return weights

    def get_class_weights_preprocessed(self):
        return torch.tensor(CLASS_WEIGHTS)

    def train_dataloader(self):
        #return self.val_dataloader()
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()
        # return DataLoader(
        #     self.test_dataset,
        #     batch_size=self.batch_size,
        #     # num_workers=self.num_workers,
        #     # persistent_workers=True,
        # )
