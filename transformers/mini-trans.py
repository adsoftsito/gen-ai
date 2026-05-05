import numpy as np

np.random.seed(42)

# -----------------------------
# 1. Datos (toy dataset)
# -----------------------------
sentences = [
    ["el", "gato", "come"],
    ["el", "perro", "come"],
    ["gato", "come", "pescado"],
    ["perro", "come", "carne"]
]

# vocabulario
vocab = list(set(word for s in sentences for word in s))
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}

vocab_size = len(vocab)

def encode(sentence):
    return [word2idx[w] for w in sentence]

data = [encode(s) for s in sentences]

# -----------------------------
# 2. Hiperparámetros
# -----------------------------
d_model = 16
seq_len = 3
lr = 0.01
epochs = 300

# -----------------------------
# 3. Parámetros
# -----------------------------
# embeddings
E = np.random.randn(vocab_size, d_model)

# attention
W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)

# feed forward
W1 = np.random.randn(d_model, d_model)
W2 = np.random.randn(d_model, d_model)

# output
W_out = np.random.randn(d_model, vocab_size)

# -----------------------------
# 4. Utilidades
# -----------------------------
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def cross_entropy(pred, target):
    return -np.log(pred[target] + 1e-9)

# -----------------------------
# 5. Positional Encoding (simple)
# -----------------------------
def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** ((2*i)/d_model)))
            if i+1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** ((2*i)/d_model)))
    return PE

PE = positional_encoding(seq_len, d_model)

# -----------------------------
# 6. Forward
# -----------------------------
def forward(x):

    X = E[x] + PE  # embeddings + posición

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.T / np.sqrt(d_model)
    attn = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    Z = attn @ V

    # feed forward
    H = np.maximum(0, Z @ W1)
    H2 = H @ W2

    logits = H2 @ W_out
    probs = np.array([softmax(l) for l in logits])

    return probs, attn, X, Z, H

# -----------------------------
# 7. Entrenamiento (simplificado)
# -----------------------------
for epoch in range(epochs):

    total_loss = 0

    for seq in data:

        probs, attn, X, Z, H = forward(seq)

        # predecir siguiente palabra
        loss = 0
        for i in range(len(seq)-1):
            loss += cross_entropy(probs[i], seq[i+1])

        total_loss += loss

        # -------- BACKPROP (simplificado) --------
        # Nota: simplificado para claridad (no completo como en PyTorch)

        grad_logits = probs.copy()

        for i in range(len(seq)-1):
            grad_logits[i][seq[i+1]] -= 1

        # salida
        dW_out = H.T @ grad_logits

        # feed forward
        dH2 = grad_logits @ W_out.T
        dW2 = H.T @ dH2

        dH = dH2 @ W2.T
        dH[H <= 0] = 0

        dW1 = Z.T @ dH

        # actualizar (sin backprop completo a attention para simplicidad)
        W_out -= lr * dW_out
        W1 -= lr * dW1
        W2 -= lr * dW2

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# -----------------------------
# 8. Prueba
# -----------------------------
test = ["el", "gato", "come"]
x = encode(test)

probs, attn, _, _, _ = forward(x)

print("\nFrase:", test)
print("Predicción siguiente palabra:")

pred_idx = np.argmax(probs[-1])
print(idx2word[pred_idx])

print("\nMatriz de atención:")
print(attn)
