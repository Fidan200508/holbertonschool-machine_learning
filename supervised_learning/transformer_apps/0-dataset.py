#!/usr/bin/env python3
"""Dataset module for Portuguese to English machine translation."""

from transformers import AutoTokenizer
from setup import load_pt2en


class Dataset:
    """Loads and tokenizes the TED Portuguese-English dataset."""

    def __init__(self):
        """Initialize the training/validation datasets and tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Create sub-word tokenizers for Portuguese and English.

        Args:
            data: tf.data.Dataset containing (pt, en) sentence pairs.

        Returns:
            tokenizer_pt: trained Portuguese tokenizer
            tokenizer_en: trained English tokenizer
        """
        tokenizer_pt_base = AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )

        tokenizer_en_base = AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def pt_iterator():
            """Yield Portuguese sentences as strings."""
            for pt, _ in data.as_numpy_iterator():
                yield pt.decode('utf-8')

        def en_iterator():
            """Yield English sentences as strings."""
            for _, en in data.as_numpy_iterator():
                yield en.decode('utf-8')

        tokenizer_pt = tokenizer_pt_base.train_new_from_iterator(
            pt_iterator(),
            vocab_size=2 ** 13
        )

        tokenizer_en = tokenizer_en_base.train_new_from_iterator(
            en_iterator(),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
