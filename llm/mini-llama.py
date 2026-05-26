import tensorflow as tf
import numpy as np
import re

# =========================================================
# MINI LLAMA GPT
# =========================================================
# Features:
# - Decoder-only Transformer
# - RMSNorm
# - SwiGLU
# - RoPE (simplified)
# - Causal Attention
# - Temperature Sampling
# - Top-K Sampling
# =========================================================

# =========================================================
# 1. LOAD DATASET
# =========================================================

with open("book.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# clean text
text = re.sub(r'[^a-z0-9\s]', '', text)

print("\n========================")
print(" DATASET PREVIEW ")
print("========================\n")

print(text[:500])

# =========================================================
# 2. TOKENIZER
# =========================================================

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters=''
)

tokenizer.fit_on_texts([text])

vocab_size = len(tokenizer.word_index) + 1

tokens = tokenizer.texts_to_sequences([text])[0]

print(f"\nVocabulary Size: {vocab_size}")
print(f"Total Tokens: {len(tokens)}")

# =========================================================
# 3. TRAINING WINDOWS
# =========================================================

SEQ_LEN = 16

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

EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256

BATCH_SIZE = 16
EPOCHS = 200

# =========================================================
# 5. RMSNorm
# =========================================================

class RMSNorm(tf.keras.layers.Layer):

    def __init__(self, dim, eps=1e-6):

        super().__init__()

        self.eps = eps

        self.scale = self.add_weight(
            shape=(dim,),
            initializer="ones",
            trainable=True
        )

    def call(self, x):

        rms = tf.sqrt(
            tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
            + self.eps
        )

        return self.scale * (x / rms)

# =========================================================
# 6. ROTARY POSITIONAL EMBEDDINGS
# =========================================================

def rotary_embedding(seq_len, dim):

    positions = np.arange(seq_len)[:, None]

    dims = np.arange(dim)[None, :]

    angles = positions / np.power(
        10000,
        (2 * (dims // 2)) / dim
    )

    emb = np.zeros((seq_len, dim))

    emb[:, 0::2] = np.sin(angles[:, 0::2])
    emb[:, 1::2] = np.cos(angles[:, 1::2])

    return tf.constant(emb, dtype=tf.float32)

rope = rotary_embedding(SEQ_LEN, EMBED_DIM)

# =========================================================
# 7. SWIGLU
# =========================================================

class SwiGLU(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_dim,
                 output_dim):

        super().__init__()

        # expand
        self.w1 = tf.keras.layers.Dense(
            hidden_dim * 2
        )

        # project back
        self.w2 = tf.keras.layers.Dense(
            output_dim
        )

    def call(self, x):

        x_proj = self.w1(x)

        x1, x2 = tf.split(
            x_proj,
            2,
            axis=-1
        )

        # swish
        swish = x1 * tf.nn.sigmoid(x1)

        gated = swish * x2

        # back to embed_dim
        return self.w2(gated)

# =========================================================
# 8. LLAMA BLOCK
# =========================================================

class LlamaBlock(tf.keras.layers.Layer):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 ff_dim):

        super().__init__()

        # attention
        self.norm1 = RMSNorm(embed_dim)

        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )

        # feedforward
        self.norm2 = RMSNorm(embed_dim)

        self.ffn = SwiGLU(
            ff_dim,
            embed_dim
        )

    # -----------------------------------------------------
    # CAUSAL MASK
    # -----------------------------------------------------

    def causal_mask(self, seq_len):

        return tf.linalg.band_part(
            tf.ones((seq_len, seq_len)),
            -1,
            0
        )

    # -----------------------------------------------------
    # FORWARD
    # -----------------------------------------------------

    def call(self, x):

        seq_len = tf.shape(x)[1]

        # =================================================
        # ATTENTION
        # =================================================

        norm_x = self.norm1(x)

        mask = self.causal_mask(seq_len)

        attn_output = self.att(
            norm_x,
            norm_x,
            attention_mask=mask
        )

        x = x + attn_output

        # =================================================
        # FEEDFORWARD
        # =================================================

        ff_out = self.ffn(
            self.norm2(x)
        )

        x = x + ff_out

        return x

# =========================================================
# 9. BUILD MODEL
# =========================================================

inputs = tf.keras.Input(shape=(SEQ_LEN,))

# ---------------------------------------------------------
# TOKEN EMBEDDINGS
# ---------------------------------------------------------

x = tf.keras.layers.Embedding(
    vocab_size,
    EMBED_DIM
)(inputs)

# ---------------------------------------------------------
# ADD ROTARY EMBEDDINGS
# ---------------------------------------------------------

x = x + rope

# ---------------------------------------------------------
# TRANSFORMER BLOCKS
# ---------------------------------------------------------

x = LlamaBlock(
    EMBED_DIM,
    NUM_HEADS,
    FF_DIM
)(x)

x = LlamaBlock(
    EMBED_DIM,
    NUM_HEADS,
    FF_DIM
)(x)

# ---------------------------------------------------------
# FINAL NORM
# ---------------------------------------------------------

x = RMSNorm(EMBED_DIM)(x)

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

outputs = tf.keras.layers.Dense(
    vocab_size,
    activation="softmax"
)(x)

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)

# =========================================================
# 10. COMPILE
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================================================
# 11. MODEL SUMMARY
# =========================================================

print("\n========================")
print(" MODEL SUMMARY ")
print("========================\n")

model.summary()

# =========================================================
# 12. TRAIN
# =========================================================

print("\n========================")
print(" TRAINING ")
print("========================\n")

model.fit(
    X,
    y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# =========================================================
# 13. TOKEN -> WORD
# =========================================================

index_to_word = {
    v:k for k,v in tokenizer.word_index.items()
}

# =========================================================
# 14. SAMPLING
# =========================================================

def sample(preds,
           temperature=0.8,
           top_k=10):

    preds = np.asarray(preds).astype("float64")

    # ----------------------------------------------
    # TEMPERATURE
    # ----------------------------------------------

    preds = np.log(preds + 1e-9) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    # ----------------------------------------------
    # TOP-K
    # ----------------------------------------------

    top_indices = np.argsort(preds)[-top_k:]

    top_probs = preds[top_indices]

    top_probs /= np.sum(top_probs)

    # ----------------------------------------------
    # SAMPLE
    # ----------------------------------------------

    return np.random.choice(
        top_indices,
        p=top_probs
    )

# =========================================================
# 15. GENERATE TEXT
# =========================================================

def generate(prompt,
             max_new_tokens=20,
             temperature=0.8):

    generated = prompt.lower()

    for _ in range(max_new_tokens):

        token_list = tokenizer.texts_to_sequences(
            [generated]
        )[0]

        token_list = token_list[-SEQ_LEN:]

        # padding
        if len(token_list) < SEQ_LEN:

            token_list = (
                [0] * (SEQ_LEN - len(token_list))
                + token_list
            )

        x_input = np.array([token_list])

        preds = model.predict(
            x_input,
            verbose=0
        )

        next_token_probs = preds[0][-1]

        next_index = sample(
            next_token_probs,
            temperature=temperature,
            top_k=5
        )

        next_word = index_to_word.get(
            next_index,
            ""
        )

        generated += " " + next_word

    return generated

# =========================================================
# 16. CHAT LOOP
# =========================================================

print("\n========================")
print(" MINI LLAMA CHAT ")
print("========================\n")

while True:

    prompt = input("You: ")

    if prompt.lower() == "exit":
        break

    result = generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8
    )

    print("\nAssistant:")
    print(result)
    print()
