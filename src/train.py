import torch
import torch.nn as nn
from tqdm import tqdm
from src.logs import get_loggers

train_logger, error_logger = get_loggers()


def train_model(model, train_loader, val_loader, epochs, lr, device, train_logger, error_logger):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_logger.info(f"Model moved to {device}")
    train_logger.info(f"Training started for {epochs} epochs")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device).float().unsqueeze(1)
            preds = model(images)

            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}")

        validate_model(model, val_loader, criterion, device, train_logger)

    torch.save(model.state_dict(), "unet_segmentation_1.pth")
    train_logger.info("Model training complete and saved successfully.")

def validate_model(model, val_loader, criterion, device, logger):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        print(f"Predictions: min={preds.min().item():.4f}, max={preds.max().item():.4f}")
        print(f"Masks: min={masks.min().item():.1f}, max={masks.max().item():.1f}")
        print(f"Loss for batch: {loss.item():.6f}")

        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device).float().unsqueeze(1)
            preds = model(images)
            loss = criterion(preds, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    logger.info(f"Validation Loss: {avg_val_loss:.4f}")
