import tensorflow as tf
import numpy as np
import re

# =========================================================
# 1. LOAD DATASET
# =========================================================

with open("book.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# clean text
text = re.sub(r'[^a-z0-9\s]', '', text)

print("\nDataset Preview:\n")
print(text[:300])

# =========================================================
# 2. TOKENIZER
# =========================================================

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters=''
)

tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

print(f"\nVocabulary Size: {total_words}")

tokens = tokenizer.texts_to_sequences([text])[0]

# =========================================================
# 3. CREATE TRAINING SEQUENCES
# =========================================================

SEQ_LEN = 10

X = []
y = []

for i in range(len(tokens) - SEQ_LEN):

    X.append(tokens[i:i+SEQ_LEN])

    y.append(tokens[i+1:i+SEQ_LEN+1])

X = np.array(X)
y = np.array(y)

print("\nTraining Shape:")
print(X.shape, y.shape)

# =========================================================
# 4. HYPERPARAMETERS
# =========================================================

EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
BATCH_SIZE = 32
EPOCHS = 100

# =========================================================
# 5. POSITIONAL EMBEDDING
# =========================================================

class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embed_dim):

        super().__init__()

        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=SEQ_LEN,
            output_dim=embed_dim
        )

    def call(self, x):

        positions = tf.range(start=0, limit=SEQ_LEN, delta=1)

        embedded_tokens = self.token_embeddings(x)

        embedded_positions = self.position_embeddings(positions)

        return embedded_tokens + embedded_positions

# =========================================================
# 6. TRANSFORMER BLOCK
# =========================================================

class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim):

        super().__init__()

        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):

        # causal mask
        batch_size = tf.shape(x)[0]

        causal_mask = tf.linalg.band_part(
            tf.ones((SEQ_LEN, SEQ_LEN)),
            -1,
            0
        )

        attention_output = self.att(
            x,
            x,
            attention_mask=causal_mask
        )

        out1 = self.layernorm1(x + attention_output)

        ffn_output = self.ffn(out1)

        return self.layernorm2(out1 + ffn_output)

# =========================================================
# 7. BUILD GPT MODEL
# =========================================================

inputs = tf.keras.Input(shape=(SEQ_LEN,))

x = PositionalEmbedding(total_words, EMBED_DIM)(inputs)

x = TransformerBlock(
    EMBED_DIM,
    NUM_HEADS,
    FF_DIM
)(x)

outputs = tf.keras.layers.Dense(total_words, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# =========================================================
# 8. COMPILE
# =========================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================================================
# 9. TRAIN
# =========================================================

model.fit(
    X,
    y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# =========================================================
# 10. GENERATION FUNCTION
# =========================================================

index_to_word = {
    v:k for k,v in tokenizer.word_index.items()
}

def sample(predictions, temperature=1.0):

    predictions = np.asarray(predictions).astype("float64")

    predictions = np.log(predictions + 1e-9) / temperature

    exp_preds = np.exp(predictions)

    predictions = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, predictions, 1)

    return np.argmax(probas)

# =========================================================
# 11. GENERATE TEXT
# =========================================================

def generate_text(seed_text,
                  next_words=20,
                  temperature=0.8):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences(
            [seed_text]
        )[0]

        token_list = token_list[-SEQ_LEN:]

        # padding
        if len(token_list) < SEQ_LEN:

            token_list = [0] * (SEQ_LEN - len(token_list)) + token_list

        token_array = np.array([token_list])

        predictions = model.predict(
            token_array,
            verbose=0
        )

        next_token_probs = predictions[0][-1]

        next_index = sample(
            next_token_probs,
            temperature
        )

        next_word = index_to_word.get(next_index, "")

        seed_text += " " + next_word

    return seed_text

# =========================================================
# 12. CHAT LOOP
# =========================================================

print("\n========================")
print(" MINI GPT TENSORFLOW ")
print("========================\n")

while True:

    prompt = input("You: ")

    if prompt.lower() == "exit":
        break

    result = generate_text(
        prompt,
        next_words=20,
        temperature=0.8
    )

    print("\nBot:")
    print(result)
    print()
