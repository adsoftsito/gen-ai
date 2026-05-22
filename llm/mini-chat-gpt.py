import numpy as np
import random

np.random.seed(42)

# =====================================================
# 1. DATASET TIPO CHAT
# =====================================================

conversations = [
    "USER hola ASSISTANT hola como estas",
    "USER quien eres ASSISTANT soy un mini chatgpt",
    "USER que es un transformer ASSISTANT es un modelo basado en atencion",
    "USER que hace un llm ASSISTANT genera texto prediciendo tokens",
    "USER que es python ASSISTANT es un lenguaje de programacion",
]

# =====================================================
# 2. TOKENIZER
# =====================================================

text = " ".join(conversations)

tokens = sorted(list(set(text.split())))

word2idx = {w:i for i,w in enumerate(tokens)}
idx2word = {i:w for w,i in word2idx.items()}

vocab_size = len(tokens)

def encode(txt):
    return [word2idx[w] for w in txt.split()]

def decode(ids):
    return " ".join(idx2word[i] for i in ids)

data = []

for conv in conversations:
    data.extend(encode(conv))

# =====================================================
# 3. HIPERPARÁMETROS
# =====================================================

d_model = 64
num_heads = 4
d_head = d_model // num_heads

context_window = 6

lr = 0.001
epochs = 1000

# =====================================================
# 4. INICIALIZACIÓN
# =====================================================

def init(shape):
    return np.random.randn(*shape) * 0.02

E = init((vocab_size, d_model))

W_Q = init((num_heads, d_model, d_head))
W_K = init((num_heads, d_model, d_head))
W_V = init((num_heads, d_model, d_head))

W_O = init((num_heads*d_head, d_model))

W1 = init((d_model, d_model*4))
W2 = init((d_model*4, d_model))

W_out = init((d_model, vocab_size))

# =====================================================
# 5. POSITIONAL ENCODING
# =====================================================

def positional_encoding(n, d):

    PE = np.zeros((n,d))

    for pos in range(n):
        for i in range(0,d,2):

            PE[pos,i] = np.sin(pos/(10000**(i/d)))

            if i+1 < d:
                PE[pos,i+1] = np.cos(pos/(10000**(i/d)))

    return PE

# =====================================================
# 6. UTILIDADES
# =====================================================

def softmax(x):

    x = x - np.max(x, axis=-1, keepdims=True)

    exp = np.exp(x)

    return exp / np.sum(exp, axis=-1, keepdims=True)

def layer_norm(x, eps=1e-5):

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    return (x - mean) / np.sqrt(var + eps)

# =====================================================
# 7. MULTI-HEAD ATTENTION
# =====================================================

def multi_head_attention(X):

    heads = []

    for h in range(num_heads):

        Q = X @ W_Q[h]
        K = X @ W_K[h]
        V = X @ W_V[h]

        scores = Q @ K.T / np.sqrt(d_head)

        # causal mask
        mask = np.tril(np.ones(scores.shape))

        scores = scores * mask - 1e9 * (1-mask)

        attn = softmax(scores)

        heads.append(attn @ V)

    concat = np.concatenate(heads, axis=-1)

    out = concat @ W_O

    return out

# =====================================================
# 8. TRANSFORMER BLOCK
# =====================================================

def transformer_block(X):

    attn = multi_head_attention(layer_norm(X))

    X = X + attn

    ff = np.maximum(0, layer_norm(X) @ W1)

    ff = ff @ W2

    X = X + ff

    return X

# =====================================================
# 9. FORWARD
# =====================================================

def forward(x):

    seq_len = len(x)

    X = E[x] + positional_encoding(seq_len, d_model)

    X = transformer_block(X)

    logits = X @ W_out

    probs = softmax(logits)

    return probs, X

# =====================================================
# 10. TRAIN SAMPLE
# =====================================================

def get_sample():

    start = random.randint(0, len(data)-context_window-2)

    x = data[start:start+context_window]

    y = data[start+1:start+context_window+1]

    return np.array(x), np.array(y)

# =====================================================
# 11. TRAINING
# =====================================================

for epoch in range(epochs):

    total_loss = 0

    for step in range(200):

        x, y = get_sample()

        probs, hidden = forward(x)

        loss = 0

        grad_logits = probs.copy()

        for i in range(len(y)):

            loss += -np.log(probs[i][y[i]] + 1e-9)

            grad_logits[i][y[i]] -= 1

        total_loss += loss

        # =========================================
        # BACKPROP SIMPLIFICADO
        # =========================================

        dW_out = hidden.T @ grad_logits

        W_out -= lr * dW_out

    if epoch % 100 == 0:
        print(f"Epoch {epoch} Loss {total_loss:.4f}")

# =====================================================
# 12. GENERACIÓN CHAT
# =====================================================

def chat(prompt, max_tokens=20):

    text = f"USER {prompt} ASSISTANT"

    x = encode(text)

    for _ in range(max_tokens):

        context = x[-context_window:]

        probs, _ = forward(context)

        next_token = np.argmax(probs[-1])

        x.append(next_token)

        word = idx2word[next_token]

        # detener si aparece USER
        if word == "USER":
            break

    return decode(x)

# =====================================================
# 13. TEST
# =====================================================

print("\n====================")
print(" MINI CHATGPT ")
print("====================\n")

while True:

    user = input("Tú: ")

    if user == "salir":
        break

    response = chat(user)

    print("\nBot:")
    print(response)
    print()
