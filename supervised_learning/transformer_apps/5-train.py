#!/usr/bin/env python3
"""Train a Transformer for Portuguese to English translation."""

import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    """Implement the Transformer learning-rate schedule."""

    def __init__(self, dm, warmup_steps=4000):
        """Initialize the learning-rate schedule."""
        super().__init__()

        self.dm = tf.cast(
            dm,
            tf.float32
        )

        self.warmup_steps = tf.cast(
            warmup_steps,
            tf.float32
        )

    def __call__(self, step):
        """Calculate the learning rate for a step."""
        step = tf.cast(
            step,
            tf.float32
        )

        arg1 = tf.math.rsqrt(step)

        arg2 = step * tf.math.pow(
            self.warmup_steps,
            -1.5
        )

        return tf.math.rsqrt(
            self.dm
        ) * tf.math.minimum(
            arg1,
            arg2
        )


def train_transformer(
    N,
    dm,
    h,
    hidden,
    max_len,
    batch_size,
    epochs
):
    """Create and train a Transformer translation model.

    Args:
        N: Number of blocks in the encoder and decoder.
        dm: Dimensionality of the model.
        h: Number of attention heads.
        hidden: Number of units in feed-forward layers.
        max_len: Maximum number of tokens per sequence.
        batch_size: Batch size used during training.
        epochs: Number of epochs to train.

    Returns:
        The trained Transformer model.
    """
    data = Dataset(
        batch_size,
        max_len
    )

    input_vocab = (
        data.tokenizer_pt.vocab_size + 2
    )

    target_vocab = (
        data.tokenizer_en.vocab_size + 2
    )

    transformer = Transformer(
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_len,
        max_len
    )

    learning_rate = CustomSchedule(
        dm,
        warmup_steps=4000
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    loss_object = (
        tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none'
        )
    )

    train_loss = tf.keras.metrics.Mean(
        name='train_loss'
    )

    train_accuracy = (
        tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy'
        )
    )

    def loss_function(real, predictions):
        """Calculate loss while ignoring padding tokens."""
        loss = loss_object(
            real,
            predictions
        )

        mask = tf.cast(
            tf.not_equal(real, 0),
            loss.dtype
        )

        loss *= mask

        return tf.math.divide_no_nan(
            tf.reduce_sum(loss),
            tf.reduce_sum(mask)
        )

    @tf.function
    def train_step(inputs, target):
        """Perform one training step."""
        target_input = target[:, :-1]
        target_real = target[:, 1:]

        (
            encoder_mask,
            combined_mask,
            decoder_mask
        ) = create_masks(
            inputs,
            target_input
        )

        with tf.GradientTape() as tape:
            predictions = transformer(
                inputs,
                target_input,
                training=True,
                encoder_mask=encoder_mask,
                look_ahead_mask=combined_mask,
                decoder_mask=decoder_mask
            )

            loss = loss_function(
                target_real,
                predictions
            )

        gradients = tape.gradient(
            loss,
            transformer.trainable_variables
        )

        optimizer.apply_gradients(
            zip(
                gradients,
                transformer.trainable_variables
            )
        )

        train_loss.update_state(loss)

        accuracy_mask = tf.cast(
            tf.not_equal(
                target_real,
                0
            ),
            tf.float32
        )

        train_accuracy.update_state(
            target_real,
            predictions,
            sample_weight=accuracy_mask
        )

    for epoch in range(epochs):
        train_loss.reset_state()
        train_accuracy.reset_state()

        for batch, (
            inputs,
            target
        ) in enumerate(data.data_train):

            train_step(
                inputs,
                target
            )

            if batch % 50 == 0:
                print(
                    "Epoch {}, Batch {}: Loss {}, "
                    "Accuracy {}".format(
                        epoch + 1,
                        batch,
                        train_loss.result().numpy(),
                        train_accuracy.result().numpy()
                    )
                )

        print(
            "Epoch {}: Loss {}, Accuracy {}".format(
                epoch + 1,
                train_loss.result().numpy(),
                train_accuracy.result().numpy()
            )
        )

    return transformer
