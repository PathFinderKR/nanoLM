# scripts/generate.py

import argparse
import os
import sys
import logging
from utils import load_config, set_logging, set_seed, configure_device, initialize_tokenizer, initialize_model


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.abspath(os.path.join(script_dir, '..', 'config.yaml'))

    parser = argparse.ArgumentParser(description="Generate text using the trained model.")
    parser.add_argument(
        '--config',
        type=str,
        default=default_config_path,
        help=f'Path to the configuration YAML file. Defaults to {default_config_path}'
    )
    return parser.parse_args()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))

    # Parse command-line arguments
    args = parse_args()

    # Load the configuration file
    config = load_config(args.config)
    required_config_keys = ['model', 'tokenizer', 'generation']
    required_nested_keys = {
        'model': ['name', 'arch', 'context_size', 'n_layer', 'n_head', 'n_embed'],
        'tokenizer': ['vocab_path'],
        'generation': ['model_path', 'prompt', 'max_new_tokens', 'temperature']
    }
    for key in required_config_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")
        for nested_key in required_nested_keys.get(key, []):
            if nested_key not in config[key]:
                raise ValueError(f"Missing required key '{nested_key}' in configuration section '{key}'.")

    # Set logging
    set_logging(os.path.join(root_dir, config['generation']['log_file']))
    # Set the random seed for reproducibility
    set_seed(config['generation'].get('seed', 42))  # Default to 42
    # Automatic device configuration
    device = configure_device()

    # Load the tokenizer
    tokenizer = initialize_tokenizer(config, root_dir)

    # Initialize model and load the checkpoint
    model = initialize_model(config, device)
    model.load_checkpoint(os.path.join(root_dir, config['generation']['model_path']))

    # Generate text
    try:
        generated_text = model.generate(
            tokenizer=tokenizer,
            prompt=config['generation']['prompt'],
            max_new_tokens=config['generation']['max_new_tokens'],
            temperature=config['generation']['temperature'],
            device=device
        )
    except Exception as e:
        logging.error(f"An error occurred during text generation: {e}")
        sys.exit(1)

    print("=== Generated Text ===")
    print(generated_text)
    print("=======================")


if __name__ == "__main__":
    main()
