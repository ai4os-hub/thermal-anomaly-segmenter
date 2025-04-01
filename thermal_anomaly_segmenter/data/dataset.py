"""
Dataset class for semantic segmentation
"""
import io
import json
import os
import secrets

import cv2
import lz4.frame as lz4frame
import numpy as np
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    PyTorch data for training a binary semantic segmentation model for
    anomaly detection.
    """

    CLASSES = ['background', 'thermal']

    def __init__(
            self,
            dataset_path: str,
            split: str = "train",
            transform=None,
            target_transform=None,
            augmentation=None,
            pixel_threshold: int = None,
            repeat_images: int = 1,
            augmentation_range: tuple = (-2, 2)
    ):
        """
        Initialize the data

        :param dataset_path: path to the folder containing the data
          (must be in a specific format created with dataset_creation.py)
        :param split: name of the split to use (train, val, test)
        :param transform: pytorch transform to apply to the images
        :param target_transform: pytorch transform to apply to the label masks
        :param augmentation: albumentations augment to apply to images & masks
        :param pixel_threshold: minimum number of foreground pixels in a mask
         for the image to be included in the data
        :param repeat_images: number of times to repeat images in the data
        :param augmentation_range: range of uniform distribution to draw from
         for augmenting the thermal images
        """

        self.dataset_path = dataset_path
        self.image_path = os.path.join(dataset_path, "image")
        self.mask_path = os.path.join(dataset_path, "label")

        self.mask_file_extension = ".png"

        self.augmentation = augmentation
        self.augmentation_range = augmentation_range

        # Read dataset_info.json file in dataset_path
        with open(os.path.join(dataset_path, "dataset_info.json"), "r") as f:
            dataset_info = json.load(f)

        self.image_file_extension = ".npy.lz4"
        default_splits = dataset_info["default_splits"]
        if split is None:
            subsets = dataset_info["subsets"]
        else:
            subsets = default_splits[split]

        file_list = []
        # Iterate over all subset folders and save relative path to file_list
        for subset in subsets:
            for f in os.listdir(os.path.join(dataset_path, "label", subset)):
                if not self.file_is_hidden(f):
                    file_list.append(
                        os.path.join(subset, os.path.splitext(f)[0])
                    )  # Append base name without extension

        self.ids = file_list

        if pixel_threshold is not None:
            self.ids = self.filter_ids(self.ids, pixel_threshold)

        original_ids = self.ids.copy()
        if repeat_images > 1:
            for i in range(1, repeat_images):
                self.ids.extend(original_ids)

        self.images_fps = [
            os.path.join(self.image_path, image_id + self.image_file_extension)
            for image_id in self.ids
        ]
        self.masks_fps = [
            os.path.join(self.mask_path, image_id + self.mask_file_extension)
            for image_id in self.ids
        ]

        classes = ['thermal']
        self.class_values = [
            self.CLASSES.index(cls.lower()) for cls in classes
        ]

        self.transform = transform
        self.target_transform = target_transform

        self.id2label = {1: 'thermal'}
        self.label2id = {'thermal': 1}

        print(f"Images to load in '{split}': {len(self.ids)}")

    def __len__(self):
        """Get data length
        """
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Get an item from the data

        :param idx: index of the item to get
        :return: tuple or dictionary containing the image, mask,
          original image and original mask
        """

        # Read numpy array image
        with open(self.images_fps[idx], 'rb') as f:
            compressed_data = f.read()
            decompressed_data = lz4frame.decompress(compressed_data)
            numpy_data = io.BytesIO(decompressed_data)
            image = np.load(numpy_data, allow_pickle=True).astype(np.float32)

        # Read the mask
        mask = cv2.imread(self.masks_fps[idx], 0)

        original_image = image.copy()
        original_mask = mask.copy()

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype(np.float32).round()

        # If augmentation is enabled, apply it
        if self.augmentation:
            augment_lower, augment_upper = self.augmentation_range
            image += secrets.SystemRandom().uniform(augment_lower,
                                                    augment_upper)

            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask, original_image, original_mask

    def file_is_hidden(self, p):
        """
        Helper function to check if a file is hidden

        :param p: path to the file (filename)
        """
        return p.startswith('.')  # linux-osx

    def filter_ids(self, ids, pixel_threshold):
        """
        Filter the ids based on the number of foreground pixels in the mask

        :param ids: list of image ids
        :param pixel_threshold: minimum number of foreground pixels in the mask
        :return: filtered list of image ids
        """

        new_ids = []
        for id in ids:
            path = os.path.join(self.mask_path, id + self.mask_file_extension)
            mask = cv2.imread(path, 0).round()

            # Count pixels with value 1
            pixel_count = np.count_nonzero(mask == 1)

            if (pixel_count > pixel_threshold):
                new_ids.append(id)

        return new_ids

    def get_mean_std(self):
        """Calculate the statistics for the data"""

        # Calculate mean and std for unmasked images
        n = 0
        psum = None  # np.zeros(3)  # Initialize sum for each channel
        psum_sq = None  # np.zeros(3)  # Initialize sum of squares for each ch
        channel_count = None

        for image_path in self.images_fps:

            # Load image
            with open(image_path, 'rb') as f:
                compressed_data = f.read()
                decompressed_data = lz4frame.decompress(compressed_data)
                buffer = io.BytesIO(decompressed_data)
                image = np.load(buffer, allow_pickle=True)

            if channel_count is None:
                channel_count = image.shape[2]
                psum = np.zeros(channel_count)  # Initialize sum for each ch
                psum_sq = np.zeros(channel_count)  # Initialize sum of squares

            # Add number of pixels in image
            n += image.size // channel_count

            # Calculate sum and sum of squares for each channel separately
            psum += np.sum(image, axis=(0, 1))
            psum_sq += np.sum(image ** 2, axis=(0, 1))

        mean = psum / n
        std = np.sqrt(psum_sq / n - mean ** 2)

        return mean, std
