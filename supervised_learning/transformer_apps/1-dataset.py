#!/usr/bin/env python3
"""Dataset module for Portuguese to English machine translation."""

import transformers
from setup import load_pt2en


class Dataset:
    """Load and prepare a Portuguese-English translation dataset."""

    def __init__(self):
        """Initialize the training and validation datasets and tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English sub-word tokenizers.

        Args:
            data: Dataset containing Portuguese-English sentence pairs.

        Returns:
            tokenizer_pt: Portuguese tokenizer.
            tokenizer_en: English tokenizer.
        """
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        def portuguese_iterator():
            """Yield Portuguese sentences from the dataset."""
            for pt, _ in data.as_numpy_iterator():
                yield pt.decode('utf-8')

        def english_iterator():
            """Yield English sentences from the dataset."""
            for _, en in data.as_numpy_iterator():
                yield en.decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            portuguese_iterator(),
            vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            english_iterator(),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode Portuguese and English sentences into tokens.

        Args:
            pt: Portuguese sentence tensor.
            en: English sentence tensor.

        Returns:
            pt_tokens: List of Portuguese token IDs.
            en_tokens: List of English token IDs.
        """
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(
            pt_sentence,
            add_special_tokens=False
        )
        en_tokens = self.tokenizer_en.encode(
            en_sentence,
            add_special_tokens=False
        )

        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_tokens = (
            [pt_vocab_size] +
            pt_tokens +
            [pt_vocab_size + 1]
        )
        en_tokens = (
            [en_vocab_size] +
            en_tokens +
            [en_vocab_size + 1]
        )

        return pt_tokens, en_tokens
