#!/usr/bin/env python3
"""Transformer model for Portuguese to English machine translation."""

import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """Create sinusoidal positional encodings."""
    positions = tf.cast(
        tf.range(max_seq_len)[:, tf.newaxis],
        tf.float32
    )
    dimensions = tf.cast(
        tf.range(dm)[tf.newaxis, :],
        tf.float32
    )

    angle_rates = 1.0 / tf.pow(
        10000.0,
        (
            2.0 * tf.floor(dimensions / 2.0)
        ) / tf.cast(dm, tf.float32)
    )

    angles = positions * angle_rates

    even_mask = tf.equal(
        tf.math.floormod(tf.range(dm), 2),
        0
    )

    encoding = tf.where(
        even_mask,
        tf.sin(angles),
        tf.cos(angles)
    )

    return encoding[tf.newaxis, ...]


def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate scaled dot-product attention."""
    scores = tf.matmul(q, k, transpose_b=True)

    depth = tf.cast(
        tf.shape(k)[-1],
        tf.float32
    )

    scores = scores / tf.math.sqrt(depth)

    if mask is not None:
        scores += tf.cast(mask, tf.float32) * -1e9

    weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(weights, v)

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Implement multi-head attention."""

    def __init__(self, dm, h):
        """Initialize multi-head attention."""
        super().__init__()

        if dm % h != 0:
            raise ValueError("dm must be divisible by h")

        self.dm = dm
        self.h = h
        self.depth = dm // h

        self.wq = tf.keras.layers.Dense(dm)
        self.wk = tf.keras.layers.Dense(dm)
        self.wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the model dimension into attention heads."""
        x = tf.reshape(
            x,
            (batch_size, -1, self.h, self.depth)
        )

        return tf.transpose(
            x,
            perm=[0, 2, 1, 3]
        )

    def call(self, q, k, v, mask=None):
        """Perform multi-head attention."""
        batch_size = tf.shape(q)[0]

        q = self.split_heads(
            self.wq(q),
            batch_size
        )
        k = self.split_heads(
            self.wk(k),
            batch_size
        )
        v = self.split_heads(
            self.wv(v),
            batch_size
        )

        attention, weights = scaled_dot_product_attention(
            q,
            k,
            v,
            mask
        )

        attention = tf.transpose(
            attention,
            perm=[0, 2, 1, 3]
        )

        attention = tf.reshape(
            attention,
            (batch_size, -1, self.dm)
        )

        output = self.linear(attention)

        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """Represent one Transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the encoder block."""
        super().__init__()

        self.mha = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training=False, mask=None):
        """Run the encoder block forward pass."""
        attention, _ = self.mha(
            x,
            x,
            x,
            mask=mask
        )

        attention = self.dropout1(
            attention,
            training=training
        )

        out1 = self.layernorm1(
            x + attention
        )

        feed_forward = self.dense_hidden(out1)
        feed_forward = self.dense_output(feed_forward)

        feed_forward = self.dropout2(
            feed_forward,
            training=training
        )

        return self.layernorm2(
            out1 + feed_forward
        )


class Encoder(tf.keras.layers.Layer):
    """Implement the Transformer encoder."""

    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        input_vocab,
        max_seq_len,
        drop_rate=0.1
    ):
        """Initialize the Transformer encoder."""
        super().__init__()

        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(
            input_vocab,
            dm
        )

        self.positional_encoding = positional_encoding(
            max_seq_len,
            dm
        )

        self.blocks = [
            EncoderBlock(
                dm,
                h,
                hidden,
                drop_rate
            )
            for _ in range(N)
        ]

        self.dropout = tf.keras.layers.Dropout(
            drop_rate
        )

    def call(self, x, training=False, mask=None):
        """Run the encoder forward pass."""
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)

        x *= tf.math.sqrt(
            tf.cast(self.dm, tf.float32)
        )

        x += self.positional_encoding[
            :,
            :seq_len,
            :
        ]

        x = self.dropout(
            x,
            training=training
        )

        for block in self.blocks:
            x = block(
                x,
                training=training,
                mask=mask
            )

        return x


class DecoderBlock(tf.keras.layers.Layer):
    """Represent one Transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the decoder block."""
        super().__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(
        self,
        x,
        encoder_output,
        training=False,
        look_ahead_mask=None,
        padding_mask=None
    ):
        """Run the decoder block forward pass."""
        attention1, _ = self.mha1(
            x,
            x,
            x,
            mask=look_ahead_mask
        )

        attention1 = self.dropout1(
            attention1,
            training=training
        )

        out1 = self.layernorm1(
            x + attention1
        )

        attention2, _ = self.mha2(
            out1,
            encoder_output,
            encoder_output,
            mask=padding_mask
        )

        attention2 = self.dropout2(
            attention2,
            training=training
        )

        out2 = self.layernorm2(
            out1 + attention2
        )

        feed_forward = self.dense_hidden(out2)
        feed_forward = self.dense_output(feed_forward)

        feed_forward = self.dropout3(
            feed_forward,
            training=training
        )

        return self.layernorm3(
            out2 + feed_forward
        )


class Decoder(tf.keras.layers.Layer):
    """Implement the Transformer decoder."""

    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        target_vocab,
        max_seq_len,
        drop_rate=0.1
    ):
        """Initialize the Transformer decoder."""
        super().__init__()

        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(
            target_vocab,
            dm
        )

        self.positional_encoding = positional_encoding(
            max_seq_len,
            dm
        )

        self.blocks = [
            DecoderBlock(
                dm,
                h,
                hidden,
                drop_rate
            )
            for _ in range(N)
        ]

        self.dropout = tf.keras.layers.Dropout(
            drop_rate
        )

    def call(
        self,
        x,
        encoder_output,
        training=False,
        look_ahead_mask=None,
        padding_mask=None
    ):
        """Run the decoder forward pass."""
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)

        x *= tf.math.sqrt(
            tf.cast(self.dm, tf.float32)
        )

        x += self.positional_encoding[
            :,
            :seq_len,
            :
        ]

        x = self.dropout(
            x,
            training=training
        )

        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask
            )

        return x


class Transformer(tf.keras.Model):
    """Implement the complete Transformer model."""

    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_seq_input,
        max_seq_target,
        drop_rate=0.1
    ):
        """Initialize the Transformer model."""
        super().__init__()

        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate
        )

        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate
        )

        self.linear = tf.keras.layers.Dense(
            target_vocab
        )

    def call(
        self,
        inputs,
        target,
        training=False,
        encoder_mask=None,
        look_ahead_mask=None,
        decoder_mask=None
    ):
        """Run the Transformer forward pass."""
        encoder_output = self.encoder(
            inputs,
            training=training,
            mask=encoder_mask
        )

        decoder_output = self.decoder(
            target,
            encoder_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_mask
        )

        return self.linear(decoder_output)
