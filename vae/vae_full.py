import numpy as np

np.random.seed(0)

# =========================
# Dataset binario completo
# =========================

X = np.array([
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1]
], dtype=float)

# =========================
# Inicialización
# =========================

W_mu = np.random.randn(2,3)*0.1
W_logvar = np.random.randn(2,3)*0.1
W_dec = np.random.randn(3,2)*0.5

lr = 0.02
epochs = 20000

# =========================
# Funciones
# =========================

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(y):
    return y*(1-y)

# Binary Cross Entropy
def bce(x_hat, x):
    return -np.sum(
        x*np.log(x_hat+1e-8) + (1-x)*np.log(1-x_hat+1e-8)
    )

# =========================
# Entrenamiento
# =========================

for epoch in range(epochs):

    total_loss = 0

    # KL annealing (clave)
    if epoch < 1000:
        beta = 0.0
    else:
        beta = 0.001

    for x in X:

        # -------- Encoder --------
        mu = W_mu @ x
        logvar = W_logvar @ x

        # ruido controlado
        sigma = 0.3 * np.exp(0.5*logvar)
        eps = np.random.randn(2)

        z = mu + sigma * eps

        # -------- Decoder --------
        h = W_dec @ z
        x_hat = sigmoid(h)

        # -------- Loss --------
        recon_loss = bce(x_hat, x)

        kl_loss = -0.5*np.sum(
            1 + logvar - mu**2 - np.exp(logvar)
        )

        loss = recon_loss + beta * kl_loss
        total_loss += loss

        # -------- Backprop --------
        
        # BCE gradient
        d_recon = (x_hat - x) / ((x_hat*(1-x_hat))+1e-8)
        
        delta = d_recon * dsigmoid(x_hat)

        grad_W_dec = np.outer(delta, z)

        dz = W_dec.T @ delta

        # encoder gradients
        dmu = dz + beta*mu

        dlogvar = dz * eps * 0.5*np.exp(0.5*logvar)*0.3 \
                  + beta*0.5*(np.exp(logvar)-1)

        grad_W_mu = np.outer(dmu, x)
        grad_W_logvar = np.outer(dlogvar, x)

        # -------- Update --------
        W_dec -= lr * grad_W_dec
        W_mu -= lr * grad_W_mu
        W_logvar -= lr * grad_W_logvar

    if epoch % 500 == 0:
        print("epoch:", epoch, "loss:", round(total_loss,4))

# =========================
# Evaluación (sin ruido)
# =========================

print("\n--- Reconstrucciones ---")

for x in X:
    mu = W_mu @ x
    z = mu  # sin ruido
    
    x_hat = sigmoid(W_dec @ z)
    
    print("Input:", x, "Output:", np.round(x_hat,3))
