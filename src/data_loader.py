import os
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, default_collate
from src.dataset import LiverSegmentationDataset
from torchvision import transforms
import torch

def custom_collate_fn(batch):
    # Remove None samples
    original_len = len(batch)
    batch = [sample for sample in batch if sample is not None]
    removed = original_len - len(batch)
    if removed > 0:
        print(f"[WARNING] Removed {removed} None samples from batch")
    return default_collate(batch) if batch else None

def get_dataloaders(image_root, mask_root, batch_size, train_logger, error_logger, val_split=0.2):
    train_logger.info("Initializing data loaders...")
    train_logger.info(f"Image Root: {image_root}")
    train_logger.info(f"Mask Root: {mask_root}")
    train_logger.info(f"Batch Size: {batch_size}")
    train_logger.info(f"Validation Split Ratio: {val_split}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    train_logger.info("Applied Transform: Resize to (256, 256) + ToTensor")

    # Initialize dataset
    full_dataset = LiverSegmentationDataset(
        image_root=image_root,
        mask_root=mask_root,
        transform=transform,
        train_logger=train_logger,
        error_logger=error_logger
    )
    
    total_samples = len(full_dataset)
    train_logger.info(f"Total Dataset Size: {total_samples}")

    indices = list(range(total_samples))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)
    train_logger.info(f"Split into {len(train_idx)} training and {len(val_idx)} validation samples")

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    train_logger.info(f"Train Loader - Total Batches: {len(train_loader)}")
    train_logger.info(f"Val Loader - Total Batches: {len(val_loader)}")

    # Sample logging - log first batch sizes
    for i, batch in enumerate(train_loader):
        if batch is None:
            train_logger.warning(f"Train Batch {i} is None. Skipping.")
            continue
        images, masks = batch
        train_logger.info(f"Sample Train Batch {i} - Input Shape: {images.shape}, Mask Shape: {masks.shape}")
        train_logger.info(f"Input Value Range: [{images.min().item():.4f}, {images.max().item():.4f}], "
                          f"Mask Unique Values: {torch.unique(masks)}")
        break  # Only log for the first batch to reduce noise

    return train_loader, val_loader
