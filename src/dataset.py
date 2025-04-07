import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from src.logs import get_loggers

train_logger, error_logger = get_loggers()

class LiverSegmentationDataset(Dataset):
    def __init__(self, image_root, mask_root, transform=None, train_logger=None, error_logger=None):
        self.image_paths = []
        self.mask_paths = []
        self.transform = transform
        self.train_logger = train_logger
        self.error_logger = error_logger

        self.train_logger.info(f"Initializing LiverSegmentationDataset")
        self.train_logger.info(f"Image root: {image_root}")
        self.train_logger.info(f"Mask root: {mask_root}")

        categories = os.listdir(image_root)

        for category in categories:
            img_dir = os.path.join(image_root, category)
            mask_dir = os.path.join(mask_root, category)

            if not os.path.exists(mask_dir):
                self.error_logger.error(f"Mask directory missing for category: {category}")
                continue

            img_filenames = sorted(os.listdir(img_dir))
            count_valid = 0

            for fname in img_filenames:
                img_path = os.path.join(img_dir, fname)
                mask_fname = fname.replace(".jpg", ".png").replace(".jpeg", ".png")
                mask_path = os.path.join(mask_dir, mask_fname)

                if not os.path.exists(mask_path):
                    self.error_logger.error(f"Missing mask for image: {img_path}")
                    continue

                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
                count_valid += 1

            self.train_logger.info(f"Category '{category}' - Loaded {count_valid}/{len(img_filenames)} images")

        self.train_logger.info(f"Total samples loaded: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        try:
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                raise ValueError("Image or mask is None")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to PIL for torchvision transforms
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
                mask = mask.squeeze(0)  # Assume grayscale, remove channel

            # Log details (first few only to avoid spamming)
            if index < 5:  # Log only first few examples
                self.train_logger.info(f"Sample {index}:")
                self.train_logger.info(f"  Image Path: {image_path}")
                self.train_logger.info(f"  Mask Path: {mask_path}")
                self.train_logger.info(f"  Image Shape: {image.shape}")
                self.train_logger.info(f"  Mask Shape: {mask.shape}")
                self.train_logger.info(f"  Mask Unique Values: {torch.unique(mask)}")

            return image, mask

        except Exception as e:
            self.error_logger.error(f"Failed to load index {index}: {str(e)} | Image: {image_path}, Mask: {mask_path}")
            return None
