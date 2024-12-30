# scripts/utils.py

import logging
import random
import numpy as np
import os
import yaml
import torch
from typing import Dict, Any
from tokenizer import CharTokenizer
from models.nanoGPT import nanoGPT, GPTConfig


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


def initialize_tokenizer(config: dict, root_dir: str, tokenizer_type: str):
    """
    Initializes the tokenizer by loading the vocabulary from a file or building it if not present.

    Args:
        config (dict): Configuration parameters.
        root_dir (str): The root directory of the repository.
        tokenizer_type (str): The type of tokenizer to initialize (e.g., 'char', 'bpe').

    Returns:
        Tokenizer: The initialized tokenizer.
    """
    tokenizer_classes = {
        'char': CharTokenizer,
        # 'bpe': BpeTokenizer
    }

    if tokenizer_type not in tokenizer_classes:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    tokenizer_class = tokenizer_classes[tokenizer_type]
    tokenizer = tokenizer_class()

    vocab_path = os.path.join(
        root_dir,
        config['tokenizer'].get('vocab_path', f'tokenizer_vocab_{tokenizer_type}.json')
    )

    if os.path.exists(vocab_path):
        tokenizer.load_vocab(vocab_path)
    else:
        train_path = os.path.join(root_dir, config['data']['train_path'])
        train_text = load_text(train_path)
        tokenizer.build_vocab(train_text)
        tokenizer.save_vocab(vocab_path)

    logging.info(f"Initialized {tokenizer_type} tokenizer with vocab size {tokenizer.vocab_size}.")
    return tokenizer


def initialize_model(config: dict, device: torch.device, model_type: str) -> torch.nn.Module:
    """
    Initializes the model based on the specified type.

    Args:
        config (dict): Configuration parameters.
        device (torch.device): The device to run the model on.
        model_type (str): The type of model to initialize (e.g., 'nanoGPT', 'GPT2').

    Returns:
        torch.nn.Module: The initialized model.
    """
    model_classes = {
        'nanoGPT': nanoGPT
        # 'GPT2': GPT2,
        # 'MEGABYTE': MegaByte,
        # 'N-gram': NGram
        # Add other model classes here
    }

    if model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_class = model_classes[model_type]

    model_config = GPTConfig(
        vocab_size=config['model']['vocab_size'],
        context_size=config['model']['context_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        n_embed=config['model']['n_embed'],
        dropout=config['model']['dropout']
    )

    model = model_class(model_config).to(device)
    logging.info(f"Initialized {model_type} with {model.num_parameters()} trainable parameters.")
    return model


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.SequentialLR,
                    scaler: torch.cuda.amp.GradScaler, step: int, loss: float,
                    checkpoint_path: str):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved at step {step} to {checkpoint_path}.")


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.SequentialLR,
                    scaler: torch.cuda.amp.GradScaler, checkpoint_path: str,
                    device: torch.device) -> Dict[str, Any]:
    if not os.path.isfile(checkpoint_path):
        logging.error(f"Checkpoint file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    except KeyError as e:
        logging.error(f"Missing key in checkpoint: {e}")
        raise

    step = checkpoint.get('step', None)
    loss = checkpoint.get('loss', None)

    if step is None or loss is None:
        logging.warning(f"Checkpoint loaded from {checkpoint_path} is missing 'step' or 'loss' information.")

    logging.info(f"Checkpoint loaded from {checkpoint_path} (step: {step}, loss: {loss}).")
    return checkpoint
