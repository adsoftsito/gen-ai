import numpy as np
np.random.seed(0)

# -----------------------------
# 1. Dataset
# -----------------------------
sentences = [
    "el gato come pescado",
    "el perro come carne",
    "el gato duerme mucho",
    "el perro ladra fuerte"
]

tokens = list(set(" ".join(sentences).split()))
word2idx = {w:i for i,w in enumerate(tokens)}
idx2word = {i:w for w,i in word2idx.items()}
vocab_size = len(tokens)

def encode(text):
    return [word2idx[w] for w in text.split()]

def decode(ids):
    return " ".join(idx2word[i] for i in ids)

data = [encode(s) for s in sentences]

# -----------------------------
# 2. Hiperparámetros
# -----------------------------
d_model = 32
num_heads = 2
d_head = d_model // num_heads
lr = 0.005
epochs = 400

def init(shape):
    return np.random.randn(*shape) * 0.1

# -----------------------------
# 3. Parámetros
# -----------------------------
E = init((vocab_size, d_model))

W_Q = init((num_heads, d_model, d_head))
W_K = init((num_heads, d_model, d_head))
W_V = init((num_heads, d_model, d_head))

W_O = init((num_heads * d_head, d_model))

W1 = init((d_model, d_model*2))
W2 = init((d_model*2, d_model))

W_out = init((d_model, vocab_size))

# -----------------------------
# 4. LayerNorm
# -----------------------------
def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

# -----------------------------
# 5. Positional Encoding
# -----------------------------
def positional_encoding(n, d):
    PE = np.zeros((n, d))
    for pos in range(n):
        for i in range(0, d, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i/d)))
            if i+1 < d:
                PE[pos, i+1] = np.cos(pos / (10000 ** (i/d)))
    return PE

# -----------------------------
# 6. Softmax estable
# -----------------------------
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)

# -----------------------------
# 7. Multi-Head Attention
# -----------------------------
def multi_head_attention(X):
    heads = []

    for h in range(num_heads):
        Q = X @ W_Q[h]
        K = X @ W_K[h]
        V = X @ W_V[h]

        scores = Q @ K.T / np.sqrt(d_head)

        mask = np.tril(np.ones(scores.shape))
        scores = scores * mask - 1e9 * (1 - mask)

        attn = softmax(scores)
        heads.append(attn @ V)

    concat = np.concatenate(heads, axis=-1)
    return concat @ W_O

# -----------------------------
# 8. Transformer Block
# -----------------------------
def transformer_block(X):
    # Attention + residual
    attn_out = multi_head_attention(layer_norm(X))
    X = X + attn_out

    # FeedForward + residual
    ff = np.maximum(0, layer_norm(X) @ W1)
    ff = ff @ W2
    X = X + ff

    return X

# -----------------------------
# 9. Forward
# -----------------------------
def forward(x):
    PE = positional_encoding(len(x), d_model)
    X = E[x] + PE

    X = transformer_block(X)

    logits = X @ W_out
    probs = softmax(logits)

    return probs

# -----------------------------
# 10. Entrenamiento (simplificado)
# -----------------------------
for epoch in range(epochs):
    loss_total = 0

    for seq in data:
        probs = forward(seq)

        loss = 0
        for i in range(len(seq)-1):
            loss += -np.log(probs[i][seq[i+1]] + 1e-9)

        loss_total += loss

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss {loss_total:.4f}")

# -----------------------------
# 11. Generación
# -----------------------------
def generate(text, max_len=5):
    x = encode(text)

    for _ in range(max_len):
        probs = forward(x)
        next_token = np.argmax(probs[len(x)-1])
        x.append(next_token)

    return decode(x)

# -----------------------------
# 12. Test
# -----------------------------
print(generate("el gato"))
print(generate("el perro"))
