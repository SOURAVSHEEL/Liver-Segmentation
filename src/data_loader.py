import os
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from dataset import LiverSegmentationDataset
from torchvision import transforms

def get_dataloaders(image_root, mask_root, batch_size, train_logger, error_logger, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    full_dataset = LiverSegmentationDataset(
        image_root=image_root,
        mask_root=mask_root,
        transform=transform,
        train_logger=train_logger,
        error_logger=error_logger
    )

    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_logger.info(f"Dataset split into {len(train_idx)} training and {len(val_idx)} validation samples")
    return train_loader, val_loader
