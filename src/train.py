import os
import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, dataloader, optimizer, criterion, device, epochs, train_logger, error_logger, save_dir="outputs/checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    train_logger.info(f"Model moved to {device}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        train_logger.info(f"Epoch {epoch+1}/{epochs} started")

        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            try:
                images, masks = data
                images = images.to(device)
                masks = masks.to(device).unsqueeze(1).float()

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            except Exception as e:
                error_logger.error(f"Training error in batch {i}: {str(e)}")

        avg_loss = epoch_loss / len(dataloader)
        train_logger.info(f"Epoch [{epoch+1}/{epochs}] completed. Avg Loss: {avg_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(save_dir, f"unet_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        train_logger.info(f"Checkpoint saved: {checkpoint_path}")
