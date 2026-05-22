import numpy as np
import random
import re

np.random.seed(42)

# =========================================================
# 1. CARGAR LIBRO
# =========================================================

with open("libro.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# limpiar caracteres raros
text = re.sub(r'[^a-záéíóúñü0-9\s]', '', text)

# =========================================================
# 2. TOKENIZER
# =========================================================

words = text.split()

vocab = sorted(list(set(words)))

word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}

vocab_size = len(vocab)

tokens = [word2idx[w] for w in words]

print(f"\nVocab Size: {vocab_size}")
print(f"Total Tokens: {len(tokens)}")

# =========================================================
# 3. HIPERPARÁMETROS
# =========================================================

d_model = 64
context_window = 12

lr = 0.001
epochs = 3000

# =========================================================
# 4. INICIALIZACIÓN
# =========================================================

def init(shape):
    return np.random.randn(*shape) * 0.02

# embeddings
E = init((vocab_size, d_model))

# attention
W_Q = init((d_model, d_model))
W_K = init((d_model, d_model))
W_V = init((d_model, d_model))

# feedforward
W1 = init((d_model, d_model * 2))
W2 = init((d_model * 2, d_model))

# salida
W_out = init((d_model, vocab_size))

# =========================================================
# 5. UTILIDADES
# =========================================================

def softmax(x):

    x = x - np.max(x)

    exp = np.exp(x)

    return exp / np.sum(exp)

# =========================================================
# 6. SELF ATTENTION
# =========================================================

def self_attention(X):

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.T / np.sqrt(d_model)

    # ------------------------------------------
    # MÁSCARA CAUSAL (GPT)
    # ------------------------------------------

    mask = np.tril(np.ones(scores.shape))

    scores = scores * mask - 1e9 * (1-mask)

    attn = np.array([softmax(s) for s in scores])

    Z = attn @ V

    cache = (Q,K,V,attn,Z,X)

    return Z, cache

# =========================================================
# 7. FORWARD
# =========================================================

def forward(x):

    # embeddings
    X = E[x]

    # attention
    Z, attn_cache = self_attention(X)

    # feedforward
    H = np.maximum(0, Z @ W1)

    H2 = H @ W2

    # logits
    logits = H2 @ W_out

    probs = np.array([softmax(l) for l in logits])

    cache = (
        X,Z,H,H2,logits,probs,
        attn_cache
    )

    return probs, cache

# =========================================================
# 8. BACKWARD
# =========================================================

def backward(x,y,cache):

    global E,W_Q,W_K,W_V,W1,W2,W_out

    (
        X,Z,H,H2,logits,probs,
        attn_cache
    ) = cache

    Q,K,V,attn,Z_attn,X_attn = attn_cache

    # =====================================================
    # OUTPUT GRADIENT
    # =====================================================

    dlogits = probs.copy()

    loss = 0

    for i in range(len(y)):

        loss += -np.log(probs[i][y[i]] + 1e-9)

        dlogits[i][y[i]] -= 1

    # =====================================================
    # W_out
    # =====================================================

    dW_out = H2.T @ dlogits

    dH2 = dlogits @ W_out.T

    # =====================================================
    # W2
    # =====================================================

    dW2 = H.T @ dH2

    dH = dH2 @ W2.T

    # =====================================================
    # RELU BACKWARD
    # =====================================================

    dH[H <= 0] = 0

    # =====================================================
    # W1
    # =====================================================

    dW1 = Z.T @ dH

    dZ = dH @ W1.T

    # =====================================================
    # ATTENTION BACKWARD
    # =====================================================

    dAttn = dZ @ V.T

    dV = attn.T @ dZ

    dScores = dAttn.copy()

    dQ = dScores @ K
    dK = dScores.T @ Q

    dQ /= np.sqrt(d_model)
    dK /= np.sqrt(d_model)

    # =====================================================
    # WQ WK WV
    # =====================================================

    dW_Q = X.T @ dQ
    dW_K = X.T @ dK
    dW_V = X.T @ dV

    dX_Q = dQ @ W_Q.T
    dX_K = dK @ W_K.T
    dX_V = dV @ W_V.T

    dX = dX_Q + dX_K + dX_V

    # =====================================================
    # EMBEDDINGS UPDATE
    # =====================================================

    for i, token in enumerate(x):

        E[token] -= lr * dX[i]

    # =====================================================
    # UPDATE PESOS
    # =====================================================

    W_out -= lr * dW_out

    W2 -= lr * dW2
    W1 -= lr * dW1

    W_Q -= lr * dW_Q
    W_K -= lr * dW_K
    W_V -= lr * dW_V

    return loss

# =========================================================
# 9. SAMPLE TRAINING WINDOW
# =========================================================

def get_sample():

    start = random.randint(0, len(tokens)-context_window-2)

    x = tokens[start:start+context_window]

    y = tokens[start+1:start+context_window+1]

    return np.array(x), np.array(y)

# =========================================================
# 10. TRAINING
# =========================================================

print("\nEntrenando...\n")

for epoch in range(epochs):

    total_loss = 0

    for step in range(200):

        x,y = get_sample()

        probs,cache = forward(x)

        loss = backward(x,y,cache)

        total_loss += loss

    if epoch % 100 == 0:

        print(f"Epoch {epoch} Loss {total_loss:.4f}")

# =========================================================
# 11. SAMPLING (ANTI MODE COLLAPSE)
# =========================================================

def sample_token(probs, temperature=0.9, top_k=5):

    # ------------------------------------------
    # TEMPERATURE
    # ------------------------------------------

    probs = np.log(probs + 1e-9) / temperature

    probs = np.exp(probs)

    probs = probs / np.sum(probs)

    # ------------------------------------------
    # TOP-K SAMPLING
    # ------------------------------------------

    top_indices = np.argsort(probs)[-top_k:]

    top_probs = probs[top_indices]

    top_probs = top_probs / np.sum(top_probs)

    # ------------------------------------------
    # SAMPLE ALEATORIO
    # ------------------------------------------

    chosen = np.random.choice(top_indices, p=top_probs)

    return chosen

# =========================================================
# 12. GENERACIÓN MEJORADA
# =========================================================

def generate(prompt,
             max_tokens=30,
             temperature=0.8,
             top_k=5):

    words_prompt = prompt.lower().split()

    x = []

    for w in words_prompt:

        if w in word2idx:
            x.append(word2idx[w])

    if len(x) == 0:
        return "No conozco esas palabras"

    result = x.copy()

    used_recent = []

    for _ in range(max_tokens):

        context = result[-context_window:]

        probs,_ = forward(context)

        next_probs = probs[-1].copy()

        # --------------------------------------
        # PENALIZAR REPETICIÓN
        # --------------------------------------

        for token in used_recent[-5:]:

            next_probs[token] *= 0.3

        next_probs = next_probs / np.sum(next_probs)

        # --------------------------------------
        # SAMPLE
        # --------------------------------------

        next_token = sample_token(
            next_probs,
            temperature=temperature,
            top_k=top_k
        )

        result.append(next_token)

        used_recent.append(next_token)

    return " ".join(idx2word[i] for i in result)

# =========================================================
# 13. CHAT
# =========================================================

print("\n==========================")
print(" MINI BOOK CHATGPT ")
print("==========================\n")

while True:

    user = input("Tú: ")

    if user.lower() == "salir":
        break

    response = generate(
        user,
        max_tokens=25,
        temperature=0.8,
        top_k=5
    )

    print("\nBot:")
    print(response)
    print()
