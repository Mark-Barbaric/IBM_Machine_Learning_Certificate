import tensorflow as tf


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 rate=0.1):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim
        )
        self.ffn = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(ff_dim, activation='relu'),
                tf.keras.layers.Dense(embedding_dim)
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attention_output = self.attention(inputs)
        attention_output = self.dropout1(attention_output, training=training)
        output1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(output1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(output1 + ffn_output)


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 maxlen: int,
                 vocab_size: int,
                 embedding_dim: int):
        super().__init__()
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        )
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=maxlen,
            output_dim=embedding_dim
        )

    def call(self, inputs):
        maxlen = tf.keras.shape(inputs)[-1]
        positions = tf.keras.arange(
            start=0,
            stop=maxlen,
            step=1
        )
        position_embeddings = self.position_embedding(positions)
        token_embeddings = self.token_embedding(inputs)
        return token_embeddings + position_embeddings


class NERModel(tf.keras.layers.Layer):
    def __init__(self,
                 num_tags: int,
                 vocab_size: int,
                 maxlen: int,
                 embedding_dim=32,
                 num_heads=2,
                 ff_dim=32):
        super().__init__()
        self.text_vectorizer = tf.keras.layers.TextVectorization(
            output_sequence_length=maxlen,
            max_tokens=vocab_size
        )
        self.embedding_layer = TokenAndPositionEmbedding(
            maxlen=maxlen,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )
        self.transformer_block = TransformerBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim
        )
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.ff = tf.keras.layers.Dense(
            ff_dim,
            activation="relu"
        )
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.ff_final = tf.keras.layers.Dense(
            num_tags,
            activation="softmax"
        )

    def call(self, inputs, training=False):
        masked_inputs = tf.keras.layers.Masking(inputs)
        x = self.embedding_layer(masked_inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x