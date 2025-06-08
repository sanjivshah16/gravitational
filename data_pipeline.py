"""
Data Pipeline and Preprocessing for LRA Benchmark Tasks

This file implements the data loading, preprocessing, and batching functionality
for the Long Range Arena benchmark tasks used in the attention mechanism comparison.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import random
from typing import List, Tuple, Dict, Optional, Iterator
from datasets import load_dataset


class IMDbDataset(Dataset):
    """Dataset for IMDb text classification task."""

    def __init__(self, texts, labels, tokenizer, vocab, max_length=4096):
        """
        Args:
            texts: List of text strings
            labels: List of labels (0 or 1)
            tokenizer: Function to tokenize text
            vocab: Vocabulary object for token-to-index mapping
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize and convert to indices
        tokens = self.tokenizer(text)
        token_indices = [self.vocab[token] for token in tokens]

        # Truncate or pad to max_length
        if len(token_indices) > self.max_length:
            token_indices = token_indices[:self.max_length]
        else:
            token_indices = token_indices + [self.vocab['<pad>']] * (self.max_length - len(token_indices))

        return {
            'input_ids': torch.tensor(token_indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


class ListOpsDataset(Dataset):
    """Dataset for ListOps hierarchical reasoning task."""

    def __init__(self, expressions, results, tokenizer, vocab, max_length=2048):
        """
        Args:
            expressions: List of ListOps expressions
            results: List of expression results (0-9)
            tokenizer: Function to tokenize expressions
            vocab: Vocabulary object for token-to-index mapping
            max_length: Maximum sequence length
        """
        self.expressions = expressions
        self.results = results
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        expression = self.expressions[idx]
        result = self.results[idx]

        # Tokenize and convert to indices
        tokens = self.tokenizer(expression)
        token_indices = [self.vocab[token] for token in tokens]

        # Truncate or pad to max_length
        if len(token_indices) > self.max_length:
            token_indices = token_indices[:self.max_length]
        else:
            token_indices = token_indices + [self.vocab['<pad>']] * (self.max_length - len(token_indices))

        return {
            'input_ids': torch.tensor(token_indices, dtype=torch.long),
            'label': torch.tensor(result, dtype=torch.long)
        }


def listops_tokenizer(text):
    """Custom tokenizer for ListOps task."""
    # Split on spaces and treat special tokens as separate tokens
    tokens = []
    for token in text.split():
        if token in ['[', ']', 'MIN', 'MAX', 'MED', 'SUM_MOD']:
            tokens.append(token)
        else:
            # For numbers, treat each digit as a separate token
            tokens.extend(list(token))
    return tokens



def create_imdb_dataloaders(batch_size=32, max_length=512, subset_size=None):
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    from torchtext.datasets import IMDB
    import torch
    from torch.utils.data import DataLoader

    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for label, line in data_iter:
            yield tokenizer(line)

    # Build vocab from training data
    train_iter = IMDB(split="train")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def process_data(split):
        data_iter = IMDB(split=split)
        processed = []
        for label, text in data_iter:
            token_ids = vocab(tokenizer(text))[:max_length]
            tensor = torch.tensor(token_ids + [0] * (max_length - len(token_ids)))
            label_tensor = torch.tensor(1 if label == "pos" else 0)
            processed.append((tensor, label_tensor))
        return processed

    train_data = process_data("train")
    test_data = process_data("test")

    # ✅ Subset the data if requested
    if subset_size:
        train_data = train_data[:subset_size]
        test_data = test_data[:subset_size]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return {
        "train": train_loader,
        "test": test_loader,
        "vocab_size": len(vocab)
    }


def create_listops_dataloaders(batch_size=32, max_length=2000, subset_size=None):
    from datasets import load_dataset
    import torch
    from torch.utils.data import DataLoader

    # ✅ NEW: load official LRA ListOps
    dataset = load_dataset("lra", name="listops", split=["train", "validation"])
    vocab = set()

    def tokenize(example):
        tokens = example["input"].split()
        vocab.update(tokens)
        return {"input_ids": tokens, "label": int(example["target"])}

    tokenized_datasets = [d.map(tokenize) for d in dataset]

    vocab = {word: idx for idx, word in enumerate(sorted(vocab))}
    vocab_size = len(vocab)

    def encode(example):
        input_ids = [vocab[token] for token in example["input_ids"][:max_length]]
        return torch.tensor(input_ids), torch.tensor(example["label"])

    train_data = [encode(ex) for ex in tokenized_datasets[0]]
    val_data = [encode(ex) for ex in tokenized_datasets[1]]

    if subset_size:
        train_data = train_data[:subset_size]
        val_data = val_data[:subset_size]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_data, batch_size=batch_size)

    return {
        "train": train_loader,
        "test": test_loader,
        "vocab_size": vocab_size
    }



def create_padding_mask(batch, pad_idx=1):
    """
    Create padding mask for attention.

    Args:
        batch: Batch of input_ids [batch_size, seq_length]
        pad_idx: Index of padding token

    Returns:
        Mask tensor [batch_size, 1, 1, seq_length]
    """
    # Create mask: 1 for non-pad tokens, 0 for pad tokens
    mask = (batch != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask
