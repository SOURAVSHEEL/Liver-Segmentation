import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms
from src.logs import get_loggers
from PIL import Image

train_logger, error_logger = get_loggers()

class LiverSegmentationDataset(Dataset):
    def __init__(self,image_root, mask_root, transform=None,train_logger=None,error_logger=None):
        self.image_paths =  []
        self.mask_paths = []
        self.transform = transform
        self.train_logger = train_logger
        self.error_logger = error_logger

        categories = os.listdir(image_root)

        for  category in categories:
            img_dir = os.path.join(image_root,category)
            mask_dir = os.path.join(mask_root,category)

            if not os.path.exists(mask_dir):
                if self.error_logger:
                    self.error_logger.error(f"Mask directory  missing for category: {category}")
                continue

            for fname in sorted(os.listdir(img_dir)):
                img_path = os.path.join(img_dir,fname)
                mask_fname = fname.replace(".jpg",".png").replace(".jpeg",".png")
                mask_path = os.path.join(mask_dir,mask_fname)

                if not os.path.exists(mask_path):
                    if  self.error_logger:
                        self.error_logger.error(f"Missing mask for image: {img_path}")
                    continue

                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)

        if self.train_logger:
            self.train_logger.info(f"Loaded {len(self.image_paths)} samples from categories: {categories}")
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            if self.error_logger:
                self.error_logger.error(f"Image or mask is None at index {index} -> Image: {image_path}, Mask: {mask_path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert numpy arrays to PIL images
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        if self.transform:

            image = self.transform(image)
            mask = self.transform(mask)
            mask = mask.squeeze(0)

        return image, mask

