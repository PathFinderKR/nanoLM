# scripts/utils.py

import logging
import random
import numpy as np
import os
import yaml
from typing import Tuple
import torch
from tokenizer import CharTokenizer
from models.GPT import GPT, GPTConfig


def set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}.")


def setup_logging(log_file_path: str, log_level: int = logging.INFO):
    """
    Configures logging to write to both a file and the console.

    Args:
        log_file_path (str): Path to the log file.
        log_level (int, optional): Logging level. Defaults to logging.INFO.
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Define the format for logs
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler for logging to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info(f"Logging to file: {log_file_path}")


def configure_device() -> torch.device:
    """
    Configure the device for training.

    Returns:
        torch.device: The device to use for training.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpu = torch.cuda.device_count()
        logging.info(f"Running on {num_gpu} CUDA GPU(s)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Running on MPS")
    else:
        device = torch.device("cpu")
        logging.info("Running on CPU")
    return device


def load_config(config_path: str) -> dict:
    """
    Loads the training configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logging.info(f"Loaded configuration from {config_path}.")
    return config


def load_text(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Load and read text data from a file.

    Args:
        file_path (str): Path to the text file.
        encoding (str, optional): File encoding. Defaults to 'utf-8'.

    Returns:
        str: The content of the text file.
    """
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding=encoding) as f:
        text = f.read()

    logging.info(f"Loaded text data from {file_path} (length: {len(text)} characters).")
    return text
