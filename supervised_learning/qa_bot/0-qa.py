#!/usr/bin/env python3
"""Question answering using BERT"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Finds answer to a question from a reference text"""

    if not question or not reference:
        return None

    try:
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad"
        )

        # Load model from TF Hub
        model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

        # Tokenize inputs
        inputs = tokenizer.encode_plus(
            question,
            reference,
            add_special_tokens=True,
            return_tensors="tf",
            truncation=True,
            max_length=512
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        # Run model
        outputs = model(
            input_word_ids=input_ids,
            input_mask=attention_mask,
            segment_ids=token_type_ids
        )

        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]

        # Get most likely start & end
        start_index = tf.argmax(start_logits, axis=1).numpy()[0]
        end_index = tf.argmax(end_logits, axis=1).numpy()[0]

        # Invalid span
        if end_index < start_index:
            return None

        # Convert tokens back to text
        tokens = input_ids[0][start_index:end_index + 1]
        answer = tokenizer.decode(tokens)

        # Clean special tokens
        answer = answer.replace("[CLS]", "").replace("[SEP]", "").strip()

        return answer if answer else None

    except Exception:
        return None
