import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from dataset import LiverSegmentationDataset
from model import UNet
from train import train_model
from logs import get_loggers

def run_training_pipeline():
    train_logger, error_logger = get_loggers()

    train_logger.info("Pipeline started.")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logger.info(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Dataset & Dataloader
    dataset = LiverSegmentationDataset(
        image_root="data/image",
        mask_root="data/masks",
        transform=transform,
        train_logger=train_logger,
        error_logger=error_logger
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    train_logger.info(f"Dataloader ready: {len(dataset)} samples, batch size 2")

    # Model, Loss, Optimizer
    model = UNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_logger.info(f"Model: {model}")
    train_logger.info("Loss: BCELoss")
    train_logger.info("Optimizer: Adam (lr=1e-4)")
    train_logger.info("Training for 10 epochs")

    # Train
    train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=10,
        train_logger=train_logger,
        error_logger=error_logger
    )

    train_logger.info("Pipeline completed.")
