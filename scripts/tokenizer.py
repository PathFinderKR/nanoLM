# scripts/tokenizer.py

import os
import json
import logging
from typing import List


class CharTokenizer:
    def __init__(self, vocab: dict = None, pad_token: str = '<PAD>', unk_token: str = '<UNK>'):
        """
        Initialize the character-level tokenizer.

        Args:
            vocab (dict, optional): A pre-defined vocabulary mapping. If None, it will be built from data.
            pad_token (str, optional): Token used for padding. Defaults to '<PAD>'.
            unk_token (str, optional): Token used for unknown characters. Defaults to '<UNK>'.
        """
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.vocab = vocab or {}
        self.char2id = {}
        self.id2char = {}
        self.vocab_size = 0

        if vocab:
            self.char2id = vocab
            self.id2char = {idx: char for char, idx in self.char2id.items()}
            self.vocab_size = len(self.char2id)
        else:
            # Initialize with special tokens
            self.char2id = {self.pad_token: 0, self.unk_token: 1}
            self.id2char = {0: self.pad_token, 1: self.unk_token}
            self.vocab_size = 2

    def build_vocab(self, text: str):
        """
        Build vocabulary from the provided text.

        Args:
            text (str): The text data to build the vocabulary from.
        """
        unique_chars = sorted(set(text))
        logging.info(f"Building vocabulary from text. Number of unique characters: {len(unique_chars)}")

        for char in unique_chars:
            if char not in self.char2id:
                self.char2id[char] = self.vocab_size
                self.id2char[self.vocab_size] = char
                self.vocab_size += 1

        logging.info(f"Vocabulary built. Total vocabulary size: {self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of integer token IDs.

        Args:
            text (str): The text to encode.

        Returns:
            List[int]: The list of token IDs.
        """
        encoded = []
        for char in text:
            if char in self.char2id:
                encoded.append(self.char2id[char])
            else:
                encoded.append(self.char2id.get(self.unk_token, 1))  # Default to <UNK> token ID
        return encoded

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of integer token IDs back into a string.

        Args:
            tokens (List[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        decoded_chars = []
        for token in tokens:
            char = self.id2char.get(token, self.unk_token)
            decoded_chars.append(char)
        return ''.join(decoded_chars)

    def save_vocab(self, filepath: str):
        """
        Save the vocabulary to a JSON file.

        Args:
            filepath (str): The path to save the vocabulary file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.char2id, f, ensure_ascii=False, indent=4)
        logging.info(f"Vocabulary saved to {filepath}.")

    def load_vocab(self, filepath: str):
        """
        Load the vocabulary from a JSON file.

        Args:
            filepath (str): The path to the vocabulary file.
        """
        if not os.path.isfile(filepath):
            logging.error(f"Vocabulary file not found: {filepath}")
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            self.char2id = json.load(f)

        self.id2char = {idx: char for char, idx in self.char2id.items()}
        self.vocab_size = len(self.char2id)
        logging.info(f"Vocabulary loaded from {filepath}. Total vocabulary size: {self.vocab_size}")

    def __len__(self):
        return self.vocab_size


class BpeTokenizer:
    def __init__(self):
        pass

    def build_vocab(self, text: str):
        pass

    def encode(self, text: str) -> List[int]:
        pass

    def decode(self, tokens: List[int]) -> str:
        pass

    def save_vocab(self, filepath: str):
        pass

    def load_vocab(self, filepath: str):
        pass

    def __len__(self):
        pass
