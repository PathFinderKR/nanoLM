# scripts/train.py

import argparse
import logging
import os
from tqdm import tqdm
from typing import Tuple
from sklearn.model_selection import train_test_split
import itertools
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR
import wandb
from utils import load_config, set_logging, set_seed, configure_device, load_text
from models.GPT import GPT, GPTConfig
from tokenizer import CharTokenizer, BpeTokenizer


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.abspath(os.path.join(script_dir, '..', 'config.yaml'))

    parser = argparse.ArgumentParser(description="Train the transformer model on the Tiny Shakespeare dataset.")
    parser.add_argument(
        '--config',
        type=str,
        default=default_config_path,
        help=f'Path to the configuration YAML file. Defaults to {default_config_path}'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with reduced dataset and parameters.'
    )
    return parser.parse_args()


class TextDataset(Dataset):
    """
    A custom dataset class for language modeling tasks.

    Args:
        data (List[int]): Encoded text data as a list of token IDs.
        context_size (int): The size of each input sequence.
    """

    def __init__(self, data, context_size):
        self.data = data
        self.context_size = context_size

    def __len__(self):
        """
        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.data) - self.context_size

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing input and target tensors.
        """
        chunk = self.data[idx:idx + self.context_size]
        input_ids = torch.tensor(chunk, dtype=torch.long)
        target_ids = torch.tensor(self.data[idx + 1:idx + 1 + self.context_size], dtype=torch.long)
        return input_ids, target_ids


def prepare_data(raw_text: str, config: dict, tokenizer: CharTokenizer | BpeTokenizer) -> Tuple[DataLoader, DataLoader]:
    """
    Encodes the raw text data, splits it into training and validation sets,
    and creates DataLoaders for both.

    Args:
        raw_text (str): The raw text data to be encoded and split.
        config (dict): Configuration parameters.
        tokenizer (CharTokenizer | BpeTokenizer): The tokenizer for encoding the text.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    encoded = tokenizer.encode(raw_text)
    logging.info(f"Encoded text into {len(encoded)} tokens.")

    val_size = config['data'].get('val_size', 0.1)  # Default to 0.1
    seed = config['training'].get('seed', 42)  # Default to 42
    encoded_train, encoded_val = train_test_split(
        encoded,
        test_size=val_size,
        shuffle=True,
        random_state=seed
    )
    logging.info(f"Split data into {len(encoded_train)} training tokens and {len(encoded_val)} validation tokens.")

    train_dataset = TextDataset(encoded_train, config['model']['context_size'])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_dataset = TextDataset(encoded_val, config['model']['context_size'])
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_dataloader, val_dataloader


def initialize_tokenizer(config: dict, root_dir: str, tokenizer_type: str) -> CharTokenizer | BpeTokenizer:
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
        'GPT': GPT
        # 'MEGABYTE': MEGABYTE,
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
    logging.info(f"{model_type} initialized with {model.num_parameters()} trainable parameters.")
    return model


def setup_optimizer(config: dict, model) -> optim.Optimizer:
    """
    Sets up the optimizer based on the configuration.

    Args:
        config (dict): Configuration parameters.
        model (torch.nn.Module): The model to optimize.

    Returns:
        optim.Optimizer: The instantiated optimizer.
    """
    optimizer_type = config['training']['optimizer'].get('type', 'AdamW')  # Default to AdamW
    optimizer_params = config['training']['optimizer'].get('params', {})

    optimizer_classes = {
        'AdamW': optim.AdamW,
        'Adam': optim.Adam,
        'SGD': optim.SGD,
    }

    if optimizer_type not in optimizer_classes:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    optimizer_class = optimizer_classes[optimizer_type]
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    logging.info(f"{optimizer_type} optimizer initialized with parameters: {optimizer_params}.")
    return optimizer


def setup_scheduler(config: dict, optimizer: optim.Optimizer) -> SequentialLR | LinearLR:
    """
    Sets up the learning rate scheduler based on the configuration.

    Args:
        config (dict): Configuration parameters.
        optimizer (optim.Optimizer): The optimizer.

    Returns:
        SequentialLR | LinearLR: The instantiated scheduler.
    """
    scheduler_type = config['training']['scheduler'].get('type', 'none')  # Default to none
    warmup_ratio = config['training']['scheduler'].get('warmup_ratio', 0.0)  # Default to 0.0
    total_steps = config['training']['max_steps']
    warmup_steps = int(warmup_ratio * total_steps)
    remaining_steps = total_steps - warmup_steps

    if scheduler_type == 'cosine':
        scheduler_warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-5,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remaining_steps,
            eta_min=1e-6
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_steps]
        )
    elif scheduler_type == 'linear':
        scheduler_warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-5,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        scheduler_linear = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=remaining_steps
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_linear],
            milestones=[warmup_steps]
        )
    elif scheduler_type == 'none':
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=total_steps)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    logging.info(f"{scheduler_type} scheduler initialized with warmup ratio {warmup_ratio}.")
    return scheduler


def evaluate(model, dataloader: DataLoader, device: torch.device) -> float:
    """
    Evaluates the model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the validation data.
        device (torch.device): The device to run evaluation on.

    Returns:
        float: The average validation loss.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            loss = model.loss(inputs, targets)
            batch_size, context_size = inputs.size()
            total_loss += loss.item() * batch_size * context_size
            total_tokens += batch_size * context_size

    average_loss = total_loss / total_tokens
    return average_loss


def train(
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: SequentialLR,
        config: dict,
        device: torch.device,
        wandb_run: wandb.sdk.wandb_run.Run,
):
    """
    The main training loop for the GPT model, including validation.

    Args:
        model (torch.nn.Module): The GPT model to train.
        train_dataloader (DataLoader): DataLoader for the training data.
        val_dataloader (DataLoader): DataLoader for the validation data.
        optimizer (optim.Optimizer): The optimizer.
        scheduler (SequentialLR): The learning rate scheduler.
        config (dict): Configuration parameters.
        device (torch.device): The device to run training on.
        wandb_run (wandb.sdk.wandb_run.Run): The Weights & Biases run instance.
    """
    max_steps = config['training']['max_steps']

    checkpoint_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        config['training']['checkpoint_dir'],
        config['model']['name']
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_interval = config['training'].get('checkpoint_interval', 1000)  # Default to 1000
    validation_interval = config['training'].get('validation_interval', 1000)  # Default to 1000
    log_interval = config['training'].get('log_interval', 100)  # Default to 100

    grad_clip = config['training'].get('grad_clip', 0)  # Default to 0
    if grad_clip > 0:
        logging.info(f"Gradient clipping enabled with max norm: {grad_clip}")

    mixed_precision = config['training'].get('mixed_precision', True) and device.type == 'cuda'  # Default to True
    if mixed_precision:
        logging.info("Mixed precision training enabled.")

    # Initialize the GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    # Create an infinite iterator over the training DataLoader
    train_iterator = itertools.cycle(train_dataloader)

    # Set the model to training mode
    model.train()
    logging.info("Starting training loop.")

    progress_bar = tqdm(range(1, max_steps), desc="Training", total=max_steps, initial=1)
    for step in progress_bar:
        current_step = step + 1
        inputs, targets = next(train_iterator)
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero out the gradients
        optimizer.zero_grad()

        if mixed_precision:
            # Forward pass with autocast for mixed precision
            with torch.cuda.amp.autocast():
                loss = model.loss(inputs, targets)
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            # Unscale gradients and perform gradient clipping
            scaler.unscale_(optimizer)
            # Conditionally apply gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Optimizer step with scaled gradients
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular forward and backward pass
            loss = model.loss(inputs, targets)
            loss.backward()
            # Conditionally apply gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Scheduler step
        scheduler.step()

        # Log training loss to wandb
        wandb_run.log({"Train Loss": loss.item(), "Step": current_step})

        # Update progress bar
        progress_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Step': current_step})

        # Logging at specified intervals
        if current_step % log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            logging.info(f"Step [{current_step}/{max_steps}], Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}")
            wandb_run.log({"Learning Rate": current_lr, "Step": current_step})

            # If optimizer has additional parameters like momentum, log them
            if isinstance(optimizer, optim.SGD) or isinstance(optimizer, optim.RMSprop):
                for idx, group in enumerate(optimizer.param_groups):
                    logging.info(f"Optimizer Param Group {idx}: {group}")
                    wandb_run.log({f"Optimizer Param Group {idx}": group})

        # Perform validation at specified intervals
        if current_step % validation_interval == 0:
            val_loss = evaluate(model, val_dataloader, device)
            logging.info(f"Validation Loss at step {current_step}: {val_loss:.4f}")
            wandb_run.log({"Validation Loss": val_loss, "Step": current_step})

        # Save checkpoint at specified intervals
        if current_step % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'step_{current_step}.pth')
            model.save_checkpoint(checkpoint_path)
            logging.info(f"Checkpoint saved at {checkpoint_path}.")
            try:
                wandb_run.save(checkpoint_path)
                logging.info(f"Checkpoint {checkpoint_path} uploaded to WandB.")
            except Exception as e:
                logging.error(f"Failed to upload checkpoint to WandB: {e}")

    # Final validation after training
    val_loss = evaluate(model, val_dataloader, device)
    logging.info(f"Final Validation Loss: {val_loss:.4f}")

    logging.info("Training complete.")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))

    # Parse command-line arguments
    args = parse_args()

    # Load the configuration file
    config = load_config(args.config)
    required_config_keys = ['model', 'tokenizer', 'training', 'data']
    required_nested_keys = {
        'model': ['name'],
        'training': ['max_steps', 'batch_size'],
        'data': ['train_path'],
        'tokenizer': ['type']
    }
    for key in required_config_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")
        for nested_key in required_nested_keys.get(key, []):
            if nested_key not in config[key]:
                raise ValueError(f"Missing required key '{nested_key}' in configuration section '{key}'.")

    # Set logging
    set_logging(os.path.join(root_dir, config['training']['log_file']))
    # Set the random seed for reproducibility
    set_seed(config['training'].get('seed', 42))  # Default to 42
    # Automatic device configuration
    device = configure_device()

    # Debug mode
    if args.debug:
        config['training']['max_steps'] = 1000
        config['training']['batch_size'] = 8
        logging.info("Debug mode enabled")

    # Weights & Biases initialization
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project=config.get('wandb', {}).get('project', 'default_project'),
        entity=config.get('wandb', {}).get('entity', None),
        config=config,
        reinit=True,
        dir=root_dir
    )
    logging.info("Weights & Biases initialized.")

    # Initialize tokenizer and model
    tokenizer = initialize_tokenizer(config, root_dir, config['tokenizer']['type'])
    model = initialize_model(config, device, config['model']['arch'])

    # Load and prepare data
    raw_text = load_text(os.path.join(root_dir, config['data']['train_path']))
    train_dataloader, val_dataloader = prepare_data(raw_text, config, tokenizer)

    # Set up the optimizer and scheduler
    optimizer = setup_optimizer(config, model)
    scheduler = setup_scheduler(config, optimizer)

    # Train the model
    try:
        train(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            scheduler,
            config,
            device,
            wandb.run
        )
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        wandb.run.finish(early=True)
        raise

    wandb.finish()


if __name__ == "__main__":
    main()
