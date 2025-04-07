import torch
import torch.nn as nn
from tqdm import tqdm
from src.logs import get_loggers
import time

train_logger, error_logger = get_loggers()


def train_model(model, train_loader, val_loader, epochs, lr, device, train_logger, error_logger):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_logger.info(f"Model moved to device: {device}")
    train_logger.info(f"Loss Function: {criterion.__class__.__name__}")
    train_logger.info(f"Optimizer: {optimizer.__class__.__name__}, LR: {lr}")
    train_logger.info(f"Training started for {epochs} epochs")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_counter = 0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}]")
        for images, masks in pbar:
            try:
                images = images.to(device)
                masks = masks.to(device).float().unsqueeze(1)

                preds = model(images)

                if preds.shape != masks.shape:
                    error_logger.error(f"❌ Shape mismatch at epoch {epoch+1}, batch {batch_counter}")
                    error_logger.error(f"   Pred shape: {preds.shape}, Mask shape: {masks.shape}")
                    continue

                loss = criterion(preds, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_counter += 1

                # Uncomment below to log per-batch loss:
                train_logger.info(f"Batch {batch_counter} - Loss: {loss.item():.4f}")

                # Optional GPU memory usage logging
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
                    mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
                    train_logger.debug(f"GPU Memory - Allocated: {mem_alloc:.2f} MB | Reserved: {mem_reserved:.2f} MB")

            except Exception as e:
                error_logger.error(f"Error during training at batch {batch_counter}: {str(e)}")

        avg_loss = epoch_loss / (batch_counter or 1)
        train_logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}")

        validate_model(model, val_loader, criterion, device, train_logger)

    # Save with timestamp
    model_path = f"unet_segmentation_{int(time.time())}.pth"
    torch.save(model.state_dict(), model_path)
    train_logger.info(f"Model training complete. Saved to {model_path}")


def validate_model(model, val_loader, criterion, device, logger):
    model.eval()
    val_loss = 0.0
    batch_counter = 0

    logger.info(f"Validation started...")

    with torch.no_grad():
        for images, masks in val_loader:
            try:
                images = images.to(device)
                masks = masks.to(device).float().unsqueeze(1)

                preds = model(images)

                if preds.shape != masks.shape:
                    error_logger.error(f"Validation shape mismatch - Pred: {preds.shape}, Mask: {masks.shape}")
                    continue

                loss = criterion(preds, masks)
                val_loss += loss.item()
                batch_counter += 1

                if batch_counter <= 3:  # Limit to avoid too much logging
                    logger.info(f"Batch {batch_counter} - Preds: min={preds.min().item():.4f}, max={preds.max().item():.4f}")
                    logger.info(f"Mask: min={masks.min().item():.1f}, max={masks.max().item():.1f}")

            except Exception as e:
                error_logger.error(f"Error during validation at batch {batch_counter}: {str(e)}")

    avg_val_loss = val_loss / (batch_counter or 1)
    logger.info(f"Validation Loss: {avg_val_loss:.4f}")
