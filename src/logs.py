import os
import logging
from datetime import datetime


def get_loggers(log_dir=r"outputs\logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    train_log_path = os.path.join(log_dir, f"training_{timestamp}.log")
    error_log_path = os.path.join(log_dir, f"errors_{timestamp}.log")

    train_logger = logging.getLogger(f"train_logger_{timestamp}")
    train_logger.setLevel(logging.INFO)
    train_handler = logging.FileHandler(train_log_path)
    train_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    train_logger.addHandler(train_handler)

    error_logger = logging.getLogger(f"error_logger_{timestamp}")
    error_logger.setLevel(logging.ERROR)
    error_handler = logging.FileHandler(error_log_path)
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    error_logger.addHandler(error_handler)

    return train_logger, error_logger