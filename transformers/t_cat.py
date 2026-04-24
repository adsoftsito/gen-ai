import numpy as np

np.set_printoptions(precision=3, suppress=True)

# -----------------------------
# 1. Vocabulario
# -----------------------------
tokens = ["el", "gato", "come", "pescado"]

# -----------------------------
# 2. Embeddings simples (hechos a mano)
# -----------------------------
# (solo para entender, en la vida real se aprenden)
embeddings = {
    "el":       np.array([0.1, 0.0, 0.1, 0.0]),
    "gato":     np.array([1.0, 0.5, 0.2, 0.1]),
    "come":     np.array([0.9, 0.7, 0.3, 0.2]),
    "pescado":  np.array([0.8, 0.6, 0.4, 0.3]),
}

# Convertir a matriz X
X = np.array([embeddings[t] for t in tokens])

print("Embeddings (X):")
print(X)

# -----------------------------
# 3. Inicializar pesos Q, K, V
# -----------------------------
np.random.seed(1)

d_model = 4

W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)

# -----------------------------
# 4. Calcular Q, K, V
# -----------------------------
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# -----------------------------
# 5. Scores (QK^T)
# -----------------------------
scores = Q @ K.T

# Escalar
scores = scores / np.sqrt(d_model)

# -----------------------------
# 6. Softmax
# -----------------------------
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scores)

# -----------------------------
# 7. Salida
# -----------------------------
output = attention_weights @ V

# -----------------------------
# 8. Mostrar resultados
# -----------------------------
print("\nTokens:")
print(tokens)

print("\nAttention Weights:")
print(attention_weights)

print("\nOutput:")
print(output)

# -----------------------------
# 9. Interpretación clara
# -----------------------------
print("\nInterpretación de atención:")
for i, token in enumerate(tokens):
    print(f"\nPalabra: {token}")
    for j, other in enumerate(tokens):
        print(f"  -> atiende a {other}: {attention_weights[i][j]:.3f}")
