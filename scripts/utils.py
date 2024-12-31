# scripts/utils.py

import logging
import random
import numpy as np
import os
import yaml
import torch
from tokenizer import CharTokenizer, BpeTokenizer
from models.GPT import GPT, GPTConfig


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
    return config


def set_logging(log_file_path: str, log_level: int = logging.INFO):
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


def initialize_tokenizer(config: dict, root_dir: str) -> CharTokenizer | BpeTokenizer:
    """
    Initializes the tokenizer by loading the vocabulary from a file or building it if not present.

    Args:
        config (dict): Configuration parameters.
        root_dir (str): The root directory of the repository.

    Returns:
        Tokenizer: The initialized tokenizer.
    """
    tokenizer_classes = {
        'char': CharTokenizer,
        # 'bpe': BpeTokenizer
    }

    tokenizer_type = config['tokenizer']['type']
    if tokenizer_type not in tokenizer_classes:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    tokenizer_class = tokenizer_classes[tokenizer_type]
    tokenizer = tokenizer_class()

    vocab_path = os.path.join(
        root_dir,
        config['tokenizer'].get('vocab_path', f'{tokenizer_type}_tokenizer.json')
    )

    if os.path.exists(vocab_path):
        tokenizer.load_vocab(vocab_path)
    else:
        train_path = os.path.join(root_dir, config['data']['train_path'])
        train_text = load_text(train_path)
        tokenizer.build_vocab(train_text)
        tokenizer.save_vocab(vocab_path)

    config['model']['vocab_size'] = tokenizer.vocab_size

    logging.info(f"{tokenizer_type} tokenizer initialized with vocab size {tokenizer.vocab_size}.")
    return tokenizer


def initialize_model(config: dict, device: torch.device) -> torch.nn.Module:
    """
    Initializes the model based on the specified type.

    Args:
        config (dict): Configuration parameters.
        device (torch.device): The device to run the model on.

    Returns:
        torch.nn.Module: The initialized model.
    """
    model_classes = {
        'GPT': GPT
        # 'MEGABYTE': MEGABYTE,
        # 'N-gram': NGram
        # Add other model classes here
    }

    model_arch = config['model']['arch']
    if model_arch not in model_classes:
        raise ValueError(f"Unsupported model type: {model_arch}")

    model_class = model_classes[model_arch]

    model_config = GPTConfig(
        vocab_size=config['model']['vocab_size'],
        context_size=config['model']['context_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        n_embed=config['model']['n_embed'],
        dropout=config['model'].get('dropout', 0.1)  # Default to 0.1
    )

    model = model_class(model_config).to(device)
    logging.info(f"{model_arch} initialized with {model.num_parameters()} trainable parameters.")
    return model
