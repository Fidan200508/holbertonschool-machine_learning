#!/usr/bin/env python3
"""Module for loading and tokenizing a machine translation dataset."""

from setup import load_pt2en
from transformers import BertTokenizerFast


class Dataset:
    """Loads and prepares a Portuguese-to-English translation dataset."""

    def __init__(self):
        """Initialize the training/validation data and tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English sub-word tokenizers.

        Args:
            data: tf.data.Dataset containing (pt, en) sentence pairs.

        Returns:
            tokenizer_pt: Portuguese tokenizer.
            tokenizer_en: English tokenizer.
        """
        tokenizer_pt = BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        def portuguese_iterator():
            """Yield Portuguese sentences from the dataset."""
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def english_iterator():
            """Yield English sentences from the dataset."""
            for _, en in data:
                yield en.numpy().decode('utf-8')

        vocab_size = 2 ** 13

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            portuguese_iterator(),
            vocab_size=vocab_size
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            english_iterator(),
            vocab_size=vocab_size
        )

        return tokenizer_pt, tokenizer_en
