import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from pytorch_lightning import LightningDataModule
from src.configuration import COLOR_MAP
import random
from tqdm import tqdm

def rgb_to_class(mask):
    """
    Converts an RGB mask image to a class index mask using a predefined color map.
    Args:
        mask (PIL.Image or np.ndarray): The input RGB mask image where each unique color represents a class.
    Returns:
        np.ndarray: A 2D array of shape (H, W) where each value is the class index corresponding to the pixel's color.
    Notes:
        - Requires a global COLOR_MAP dictionary mapping RGB tuples to class indices.
        - The function assigns each pixel to the class whose RGB color in COLOR_MAP is closest (in L2 distance) to the pixel's color.
    """

    mask_np = np.array(mask)
    h, w, _ = mask_np.shape
    class_mask = np.zeros((h, w), dtype=np.int64)

    color_keys = np.array(list(COLOR_MAP.keys()))  # Shape: [N_classes, 3]
    color_vals = np.array(list(COLOR_MAP.values()))  # Shape: [N_classes]

    # Reshape mask to [H*W, 3]
    flat_mask = mask_np.reshape(-1, 3)

    # Compute L2 distances between each pixel and known class RGBs
    dists = np.linalg.norm(
        flat_mask[:, None, :] - color_keys[None, :, :], axis=2
    )  # Shape: [H*W, N_classes]

    # Find nearest color
    nearest_color_indices = np.argmin(dists, axis=1)
    class_mask = color_vals[nearest_color_indices].reshape(h, w)

    return class_mask


def class_to_rgb(class_mask):
    """
    Converts a 2D class mask to a 3-channel RGB mask using a predefined color map.

    Args:
        class_mask (np.ndarray): 2D array of shape (H, W) where each value represents a class index.

    Returns:
        np.ndarray: 3D array of shape (H, W, 3) representing the RGB mask, where each pixel's color corresponds to its class index as defined in COLOR_MAP.

    Note:
        COLOR_MAP should be a dictionary mapping RGB tuples to class indices.
    """
    rgb_mask = np.zeros((class_mask.shape[0], class_mask.shape[1], 3), dtype=np.uint8)
    for rgb, idx in COLOR_MAP.items():
        rgb_mask[class_mask == idx] = rgb
    return rgb_mask


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Initializes the dataset loader with directories for images and masks, and an optional transform.

        Args:
            images_dir (str): Path to the directory containing input images.
            masks_dir (str): Path to the directory containing mask images.
            transform (callable, optional): Optional transform to be applied on a sample.

        Attributes:
            ids (list): Sorted list of image filenames in the images directory.
            images_fps (list): List of full file paths to the images.
            masks_fps (list): List of full file paths to the corresponding mask images, 
                with filenames modified to end with '_train_color.png'.
            transform (callable, optional): Transform to be applied to the images and masks.
        """
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, f) for f in self.ids]
        self.masks_fps = [
            os.path.join(masks_dir, Path(f).stem + "_train_color.png") for f in self.ids
        ]
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.ids)

    def __getitem__(self, i):
        """
        Retrieves the image and corresponding mask at the specified index.

        Args:
            i (int): Index of the image and mask to retrieve.

        Returns:
            tuple: A tuple (img, mask) where:
                - img (Tensor): The transformed image tensor.
                - mask (Tensor): The transformed mask tensor, typically containing class indices.

        Notes:
            - If a transform is provided, it is applied to both the image and mask.
            - If no transform is provided, the image is converted to a tensor and the mask is converted from RGB to class indices.
        """
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
        """
        Initializes the data loader with the directory of images and an optional transform.

        Args:
            images_dir (str): Path to the directory containing image files.
            transform (callable, optional): Optional transform to be applied on a sample.

        Attributes:
            ids (list): Sorted list of image file names in the directory.
            images_fps (list): List of full file paths to the images.
            transform (callable, optional): Transform to be applied to each image.
        """
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, f) for f in self.ids]
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.ids)

    def __getitem__(self, i):
        """
        Retrieves the image at the specified index, applies optional transformations, and returns it as a tensor.

        Args:
            i (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: The transformed image tensor.

        Notes:
            - If a transform is provided, it is applied to the image.
            - If no transform is provided, the image is converted to a tensor using torchvision.transforms.ToTensor().
        """
        img = Image.open(self.images_fps[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        """
        Initializes the object with specified mean and standard deviation values.

        Args:
            mean (float, optional): The mean value for normalization. Defaults to 0.0.
            std (float, optional): The standard deviation value for normalization. Defaults to 0.01.
        """
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Applies Gaussian noise to the input image.

        Args:
            img (PIL.Image.Image): Input image to which noise will be added.

        Returns:
            PIL.Image.Image: Image with added Gaussian noise.

        Notes:
            - The image is converted to a NumPy array and normalized to [0, 1].
            - Gaussian noise with specified mean and standard deviation is added.
            - The resulting image is clipped to [0, 1], rescaled to [0, 255], and converted back to a PIL Image.
        """
        np_img = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(self.mean, self.std, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 1)
        return Image.fromarray((np_img * 255).astype(np.uint8))


class DualCompose:
    def __init__(self, transforms):
        """
        Initializes the class with the specified image transformations.

        Args:
            transforms (callable): A function or composition of functions that will be applied to the input data for preprocessing or augmentation.
        """
        self.transforms = transforms

    def __call__(self, image, target):
        """
        Applies a sequence of transformations to the given image and target.

        Args:
            image: The input image to be transformed.
            target: The corresponding target (e.g., mask or label) to be transformed.

        Returns:
            Tuple containing the transformed image and target.
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        """
        Initializes the object with a specified probability for flipping images during data augmentation.

        Args:
            flip_prob (float): The probability of flipping an image. Should be a value between 0 and 1.
        """
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        """
        Applies a random horizontal flip to both the input image and its corresponding target with a probability defined by `self.flip_prob`.

        Args:
            image (PIL.Image or torch.Tensor): The input image to be potentially flipped.
            target (PIL.Image or torch.Tensor): The corresponding target (e.g., segmentation mask) to be potentially flipped.

        Returns:
            Tuple: A tuple containing the (possibly flipped) image and target.
        """
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomRotation:
    def __init__(self, degrees):
        """
        Initializes the object with the specified rotation degrees.

        Args:
            degrees (float or int): The degree of rotation to be applied.
        """
        self.degrees = degrees

    def __call__(self, image, target):
        """
        Applies a random rotation to both the input image and its corresponding target (mask).

        Parameters:
            image (PIL.Image or Tensor): The input image to be rotated.
            target (PIL.Image or Tensor): The target mask to be rotated.

        Returns:
            tuple: A tuple containing the rotated image and target.

        Notes:
            - The rotation angle is randomly selected within the range specified by `self.degrees`.
            - The image is rotated using bilinear interpolation for smoother results.
            - The target is rotated using nearest neighbor interpolation to preserve label integrity.
        """
        angle = (
            random.uniform(-self.degrees, self.degrees)
            if isinstance(self.degrees, (int, float))
            else random.uniform(*self.degrees)
        )
        image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR)
        target = F.rotate(target, angle, interpolation=F.InterpolationMode.NEAREST)
        return image, target


class RandomCrop:
    def __init__(self, size):
        """
        Initializes the object with the specified size.

        Args:
            size (int or tuple): The size parameter to be used for initialization.
        """
        self.size = size

    def __call__(self, image, target):
        """
        Applies a random crop to both the input image and its corresponding target.

        Args:
            image (PIL.Image or Tensor): The input image to be cropped.
            target (PIL.Image or Tensor): The corresponding target (e.g., segmentation mask) to be cropped.

        Returns:
            Tuple[PIL.Image or Tensor, PIL.Image or Tensor]: The cropped image and target, both with the same randomly selected region.

        Note:
            The crop parameters are randomly generated but shared between the image and target to ensure spatial alignment.
        """
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.size)
        image = F.crop(image, i, j, h, w)
        target = F.crop(target, i, j, h, w)
        return image, target


class SegmentationTrainTransform:
    def __init__(self, image_size=(256, 256)):
        """
        Initializes the data loader with specified image and mask transformations.

        Args:
            image_size (tuple, optional): The target size (height, width) for cropping images and masks. Defaults to (256, 256).

        Attributes:
            image_resize (torchvision.transforms.Compose): Transformation pipeline to resize images to (512, 512).
            image_transform (torchvision.transforms.Compose): Transformation pipeline for data augmentation and normalization, including color jitter, Gaussian blur, Gaussian noise, conversion to tensor, and normalization.
            mask_resize (torchvision.transforms.Resize): Transformation to resize masks to (512, 512) using nearest neighbor interpolation.
            shared_transforms (DualCompose): Pipeline of transformations applied to both images and masks, including random horizontal flip, random crop to `image_size`, and random rotation.
        """
        self.image_resize = transforms.Compose(
            [
                transforms.Resize((256, 256)),
            ]
        )
        self.image_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                AddGaussianNoise(0.0, 0.03),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.3702, 0.4144, 0.4244],
                    [0.2503, 0.2664, 0.2831]
                ),
            ]
        )
        self.mask_resize = transforms.Resize((256, 256), interpolation=Image.NEAREST)
        self.shared_transforms = DualCompose(
            [
                RandomHorizontalFlip(flip_prob=0.8),
                RandomCrop(image_size),
                RandomRotation(degrees=0.30),
            ]
        )

    def __call__(self, img, mask):
        """
        Applies resizing, shared augmentations, and transformations to an input image and its corresponding mask.

        Args:
            img (PIL.Image or np.ndarray): The input image to be processed.
            mask (PIL.Image or np.ndarray): The corresponding segmentation mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Transformed image tensor.
                - Transformed mask tensor with class indices.
        """
        img = self.image_resize(img)
        mask = self.mask_resize(mask)

        # Apply shared augmentations
        img, mask = self.shared_transforms(img, mask)

        img = self.image_transform(img)
        mask = torch.from_numpy(rgb_to_class(mask)).long()
        return img, mask


class SegmentationValTransform:
    def __init__(self, image_size=(256, 256), normalize=True):
        """
        Initializes the data loader with specified image size and normalization options.

        Args:
            image_size (tuple, optional): The desired size (height, width) to which input images and masks will be resized. Defaults to (256, 256).
            normalize (bool, optional): Whether to apply normalization to the images using ImageNet mean and standard deviation. Defaults to True.

        Attributes:
            image_transform (torchvision.transforms.Compose): Transformation pipeline applied to input images, including resizing, conversion to tensor, and optional normalization.
            mask_resize (torchvision.transforms.Resize): Transformation for resizing segmentation masks using nearest neighbor interpolation.
        """
        if normalize:
            self.image_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.3702, 0.4144, 0.4244],
                        [0.2503, 0.2664, 0.2831]
                    ),
                ]
            )
        else:
            self.image_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ]
            )
        self.mask_resize = transforms.Resize(image_size, interpolation=Image.NEAREST)

    def __call__(self, img, mask):
        """
        Applies image and mask transformations to the input image and mask.

        Args:
            img (PIL.Image or np.ndarray): The input image to be transformed.
            mask (PIL.Image or np.ndarray): The corresponding segmentation mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Transformed image tensor.
                - Transformed mask tensor with class indices.
        """
        img = self.image_transform(img)
        mask = self.mask_resize(mask)
        mask = torch.from_numpy(rgb_to_class(mask)).long()
        return img, mask

class SegmentationTestTransform:
    """
    Transformation pipeline for preprocessing test images in semantic segmentation.

    Applies resizing, tensor conversion, and normalization using precomputed dataset statistics.
    """

    def __init__(self, image_size=(256, 256)):
        """
        Initializes the test-time transformation pipeline.

        Args:
            image_size (tuple, optional): Target image size (height, width). Defaults to (256, 256).
        """
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.3702, 0.4144, 0.4244],  # mean (per channel)
                    [0.2503, 0.2664, 0.2831]   # std (per channel)
                ),
            ]
        )

    def __call__(self, img):
        """
        Applies the transformation pipeline to the input image.

        Args:
            img (PIL.Image or np.ndarray): Input image.

        Returns:
            torch.Tensor: Normalized image tensor.
        """
        return self.image_transform(img)


class SegmentationDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for loading and preprocessing segmentation datasets.

    Handles dataset construction and provides data loaders for training, validation, and testing.
    Also includes utilities for class frequency analysis and normalization statistics computation.
    """

    def __init__(
        self,
        batch_size=8,
        data_path="data",
        train_transform=None,
        val_transform=None,
        test_transform=None,
        num_classes=12,
        num_workers=4,
    ):
        """
        Initializes the data module.

        Args:
            batch_size (int): Number of samples per batch.
            data_path (str or Path): Path to the root dataset directory.
            train_transform (callable, optional): Transform to apply to training data.
            val_transform (callable, optional): Transform to apply to validation data.
            test_transform (callable, optional): Transform to apply to test data.
            num_classes (int): Number of segmentation classes.
            num_workers (int): Number of workers for data loading.
        """
        super().__init__()
        self.batch_size = batch_size
        self.data_path = Path(data_path)
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.num_classes = num_classes
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Sets up datasets for training, validation, and testing.

        Args:
            stage (str, optional): Stage of training (e.g., 'fit', 'test', 'predict').
        """
        self.train_dataset = SegmentationDataset(
            images_dir=self.data_path / "images" / "train",
            masks_dir=self.data_path / "masks" / "train",
            transform=self.train_transform,
        )
        self.val_dataset = SegmentationDataset(
            images_dir=self.data_path / "images" / "val",
            masks_dir=self.data_path / "masks" / "val",
            transform=self.val_transform,
        )
        self.test_dataset = TestSegmentationDataset(
            images_dir=self.data_path / "images" / "test",
            transform=self.test_transform,
        )

    def get_class_counts(self):
        """
        Computes the frequency (pixel count) of each segmentation class in the training set.

        Returns:
            torch.Tensor: Tensor of shape (num_classes,) with class-wise pixel counts.
        """
        class_counts = torch.zeros(self.num_classes)

        for _, mask in tqdm(self.train_dataset, desc="Computing class frequencies"):
            if isinstance(mask, torch.Tensor):
                mask = mask.long()
            else:
                mask = torch.from_numpy(mask).long()

            for cls in range(self.num_classes):
                class_counts[cls] += (mask == cls).sum()

        return class_counts

    def compute_mean_std(self, batch_size=16, num_workers=4):
        """
        Computes mean and standard deviation of training images for normalization.

        Args:
            batch_size (int): Batch size for loading images.
            num_workers (int): Number of worker threads.

        Returns:
            tuple: A tuple (mean, std), where each is a list of per-channel values.
        """
        means = []
        stds = []

        dataset = SegmentationDataset(
            images_dir=self.data_path / "images" / "train",
            masks_dir=self.data_path / "masks" / "train",
            transform=SegmentationValTransform(image_size=(224, 244), normalize=False),
        )

        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )

        for imgs, _ in tqdm(loader, desc="Computing mean/std"):
            means.append(imgs.mean(dim=[0, 2, 3]))  # per-channel mean
            stds.append(imgs.std(dim=[0, 2, 3]))    # per-channel std

        mean = torch.stack(means).mean(dim=0)
        std = torch.stack(stds).mean(dim=0)
        return mean.tolist(), std.tolist()

    def train_dataloader(self):
        """
        Returns the training data loader.

        Returns:
            DataLoader: PyTorch DataLoader for the training set.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Returns the validation data loader.

        Returns:
            DataLoader: PyTorch DataLoader for the validation set.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """
        Returns the test data loader.

        Note:
            Currently uses the validation dataloader.

        Returns:
            DataLoader: PyTorch DataLoader for the test set.
        """
        return self.val_dataloader()