import numpy as np
np.random.seed(0)

# -----------------------------
# 1. Dataset simple
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

def decode(indices):
    return " ".join(idx2word[i] for i in indices)

data = [encode(s) for s in sentences]

# -----------------------------
# 2. Hiperparámetros
# -----------------------------
d_model = 32
lr = 0.005
epochs = 500

# -----------------------------
# 3. Inicialización (estable)
# -----------------------------
def init(shape):
    return np.random.randn(*shape) * 0.1

E = init((vocab_size, d_model))

W_Q = init((d_model, d_model))
W_K = init((d_model, d_model))
W_V = init((d_model, d_model))

W1 = init((d_model, d_model))
W2 = init((d_model, d_model))

W_out = init((d_model, vocab_size))

# -----------------------------
# 4. Positional Encoding dinámico
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
# 5. Softmax estable
# -----------------------------
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# -----------------------------
# 6. Masked Attention estable
# -----------------------------
def masked_attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(d_model)

    # máscara causal
    mask = np.tril(np.ones(scores.shape))
    scores = scores * mask - 1e9 * (1 - mask)

    # softmax estable
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    attn = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return attn @ V, attn

# -----------------------------
# 7. Forward
# -----------------------------
def forward(x):
    PE = positional_encoding(len(x), d_model)
    X = E[x] + PE

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    Z, attn = masked_attention(Q, K, V)

    H = np.maximum(0, Z @ W1)
    H2 = H @ W2

    logits = H2 @ W_out
    probs = softmax(logits)

    return probs, attn, X, Z, H

# -----------------------------
# 8. Entrenamiento
# -----------------------------
for epoch in range(epochs):

    total_loss = 0

    for seq in data:

        probs, attn, X, Z, H = forward(seq)

        grad_logits = probs.copy()
        loss = 0

        for i in range(len(seq)-1):
            loss += -np.log(probs[i][seq[i+1]] + 1e-9)
            grad_logits[i][seq[i+1]] -= 1

        total_loss += loss

        # backprop simplificado
        dW_out = H.T @ grad_logits

        dH2 = grad_logits @ W_out.T
        dW2 = H.T @ dH2

        dH = dH2 @ W2.T
        dH[H <= 0] = 0

        dW1 = Z.T @ dH

        # actualizar
        W_out -= lr * dW_out
        W1 -= lr * dW1
        W2 -= lr * dW2

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# -----------------------------
# 9. Generación
# -----------------------------
def generate(start_text, max_len=5):

    x = encode(start_text)

    for _ in range(max_len):
        probs, _, _, _, _ = forward(x)
        next_token = np.argmax(probs[len(x)-1])
        x.append(next_token)

    return decode(x)

# -----------------------------
# 10. Pruebas
# -----------------------------
print("\nGeneración:")
print(generate("el gato"))
print(generate("el perro"))

print("\nMatriz de atención ejemplo:")
probs, attn, _, _, _ = forward(encode("el gato come"))
print(attn)
