import torch
# from dataset import *
from src.data_loader import get_dataloaders
from src.train import train_model
from src.logs import get_loggers
from src.Unet_model import model

if __name__ == "__main__":
    train_logger, error_logger = get_loggers()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logger.info(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 2
    EPOCHS = 20
    LR = 1e-4
    IMAGE_ROOT = "data/image"
    MASK_ROOT = "data/masks"

    train_loader, val_loader = get_dataloaders(
        IMAGE_ROOT, MASK_ROOT, BATCH_SIZE, train_logger, error_logger
    )

    # model = UNet(in_channels=3, out_channels=1)
    train_logger.info(f"Model structure:\n{model}")

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        train_logger=train_logger,
        error_logger=error_logger
    )
