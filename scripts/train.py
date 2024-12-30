# scripts/train.py

import argparse
import logging
import os
from tqdm import tqdm
from typing import Tuple, Union
from sklearn.model_selection import train_test_split
import itertools
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR
import wandb
from utils import (
    set_seed,
    setup_logging,
    configure_device,
    load_config,
    load_text,
    initialize_tokenizer,
    initialize_model,
    save_checkpoint,
    load_checkpoint
)
from tokenizer import CharTokenizer


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
        '--model',
        type=str,
        default='nanoGPT',
        choices=['nanoGPT', 'GPT2', 'MEGABYTE', 'N-gram'],
        help='Model architecture to train on (nanoGPT, GPT2, MEGABYTE, N-gram).'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='char',
        choices=['char', 'bpe'],
        help='Tokenizer to use (char, bpe).'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=default_config_path,
        help=f'Path to the configuration YAML file. Defaults to {default_config_path}'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to resume training from a checkpoint.'
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


def prepare_data(raw_text: str, config: dict, tokenizer: CharTokenizer) -> Tuple[DataLoader, DataLoader]:
    """
    Encodes the raw text data, splits it into training and validation sets,
    and creates DataLoaders for both.

    Args:
        raw_text (str): The raw text data to be encoded and split.
        config (dict): Configuration parameters.
        tokenizer (CharTokenizer): The tokenizer for encoding the text.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    encoded = tokenizer.encode(raw_text)
    logging.info(f"Encoded text into {len(encoded)} tokens.")

    val_size = config['data'].get('val_size', 0.1)  # Default to 0.1
    seed = config['training'].get('seed', 42)
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
    logging.info(f"Optimizer: {optimizer_type} initialized with parameters: {optimizer_params}.")
    return optimizer


def setup_scheduler(config: dict, optimizer: optim.Optimizer, total_steps: int) -> SequentialLR:
    """
    Sets up the learning rate scheduler based on the configuration.

    Args:
        config (dict): Configuration parameters.
        optimizer (optim.Optimizer): The optimizer.
        total_steps (int): Total number of training steps.

    Returns:
        SequentialLR: The instantiated scheduler.
    """
    scheduler_type = config['training']['scheduler'].get('type', 'none')  # Default to none
    warmup_ratio = config['training']['scheduler'].get('warmup_ratio', 0.0)  # Default to 0.0
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
    logging.info(f"Scheduler: {scheduler_type} initialized with warmup ratio {warmup_ratio}.")
    return scheduler


def resume_training(model, optimizer: optim.Optimizer, scheduler: SequentialLR,
                    scaler: torch.cuda.amp.GradScaler, checkpoint_path: str, device: torch.device) \
        -> Union[Tuple[int, float], Tuple[int, None]]:
    """
    Optionally resumes training from a checkpoint if a path is provided.

    Args:
        model (torch.nn.Module): The model.
        optimizer (optim.Optimizer): The optimizer.
        scheduler (SequentialLR): The scheduler.
        scaler (torch.cuda.amp.GradScaler): The scaler for mixed precision.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): The device to load the checkpoint on.

    Returns:
        Union[Tuple[int, float], Tuple[int, None]]: The starting step and loss value.
    """
    if checkpoint_path:
        checkpoint = load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, device)
        start_step = checkpoint.get('step', 0) + 1  # Start from the next step after the checkpoint
        loss = checkpoint.get('loss', None)

        if loss is not None:
            logging.info(f"Resuming training from step {start_step} with loss {loss:.4f}.")
            return start_step, loss
        else:
            logging.warning(f"'loss' not found in checkpoint {checkpoint_path}.")
            return start_step, None
    return 0, None


def evaluate(model, dataloader: DataLoader, device: torch.device) -> float:
    """
    Evaluates the model on the validation dataset.

    Args:
        model (torch.nn.Module): The GPT model to evaluate.
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
        start_step: int
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
        start_step (int): The step to start training from.
    """
    max_steps = config['training']['max_steps']
    checkpoint_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        config['training']['checkpoint_dir']
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_interval = config['training'].get('checkpoint_interval', 1000)  # Default to 1000
    validation_interval = config['training'].get('validation_interval', 1000)  # Default to 1000
    log_interval = config['training'].get('log_interval', 100)  # Default to 100

    grad_clip = config['training']['grad_clip']
    mixed_precision = config['training'].get('mixed_precision', True) and device.type == 'cuda'  # Default to True

    # Initialize the GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    # Create an infinite iterator over the training DataLoader
    train_iterator = itertools.cycle(train_dataloader)

    # Initialize a tqdm progress bar for the entire training process
    progress_bar = tqdm(
        range(start_step, max_steps),
        desc="Training",
        total=max_steps - start_step,
        initial=start_step
    )

    for step in progress_bar:
        current_step = step + 1  # Adjust step count
        inputs, targets = next(train_iterator)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        if mixed_precision:
            # Forward pass with autocast for mixed precision
            with torch.cuda.amp.autocast():
                loss = model.loss(inputs, targets)
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            # Unscale gradients and perform gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Optimizer step with scaled gradients
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular forward and backward pass
            loss = model.loss(inputs, targets)
            loss.backward()
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
            logging.info(f"Performing validation at step {current_step}.")
            val_loss = evaluate(model, val_dataloader, device)
            logging.info(f"Validation Loss at step {current_step}: {val_loss:.4f}")
            wandb_run.log({"Validation Loss": val_loss, "Step": current_step})

        # Save checkpoint at specified intervals
        if current_step % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'step_{current_step}.pth')
            save_checkpoint(model, optimizer, scheduler, scaler, current_step, loss.item(), checkpoint_path)
            logging.info(f"Checkpoint saved at {checkpoint_path}.")
            try:
                wandb_run.save(checkpoint_path)
                logging.info(f"Checkpoint {checkpoint_path} uploaded to WandB.")
            except Exception as e:
                logging.error(f"Failed to upload checkpoint to WandB: {e}")

    # Final validation after training
    logging.info("Performing final validation after training.")
    val_loss = evaluate(model, val_dataloader, device)
    logging.info(f"Final Validation Loss: {val_loss:.4f}")
    wandb_run.log({"Final Validation Loss": val_loss})

    logging.info("Training finished.")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))

    # Parse command-line arguments
    args = parse_args()

    # Load the configuration file
    config = load_config(args.config)

    # Validate configuration parameters
    required_config_keys = ['model', 'training', 'data', 'tokenizer']
    for key in required_config_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")

    # Logging configuration
    log_file_path = os.path.join(root_dir, config['training']['log_file'])
    setup_logging(log_file_path)

    # Set the random seed for reproducibility
    set_seed(config['training'].get('seed', 42))  # Default to 42

    # Device configuration
    device = configure_device()

    # Debug mode adjustments
    if args.debug:
        config['training']['max_steps'] = 1000
        config['training']['batch_size'] = 8
        logging.info("Debug mode enabled")

    # Weights & Biases initialization
    wandb_project = config.get('wandb', {}).get('project', 'default_project')
    wandb_entity = config.get('wandb', {}).get('entity', None)
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config=config,
        resume='allow' if args.resume else False,
        reinit=True,
        dir=root_dir
    )
    wandb_run = wandb.run
    logging.info("Weights & Biases initialized.")

    # Initialize tokenizer
    tokenizer = initialize_tokenizer(config, root_dir, args.tokenizer)

    # Load and prepare data
    train_path = os.path.join(root_dir, config['data']['train_path'])
    raw_text = load_text(train_path)
    train_dataloader, val_dataloader = prepare_data(raw_text, config, tokenizer)

    # Update vocab_size in config based on tokenizer
    config['model']['vocab_size'] = tokenizer.vocab_size

    # Initialize the GPT model based on the selected model type
    model = initialize_model(config, device, args.model)

    # Set up the optimizer
    optimizer = setup_optimizer(config, model)

    # Set up the scheduler
    total_steps = config['training']['max_steps']
    scheduler = setup_scheduler(config, optimizer, total_steps)

    # Initialize the GradScaler for mixed precision training
    mixed_precision = config['training'].get('mixed_precision', True) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    # Optionally resume training from a checkpoint
    if args.resume:
        start_step, loss = resume_training(model, optimizer, scheduler, scaler, args.resume, device)
    else:
        start_step, loss = 0, None

    config['training']['start_step'] = start_step

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
            wandb_run,
            start_step
        )
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        wandb_run.finish(early=True)
        raise

    wandb.finish()
    logging.info("Training completed successfully.")


if __name__ == "__main__":
    main()
