import numpy as np

np.random.seed(0)

# =========================
# Datos
# =========================

x = np.array([1.0,0.0,1.0])

# =========================
# Inicialización de pesos
# =========================

W_mu = np.random.randn(2,3)*0.1
W_logvar = np.random.randn(2,3)*0.1
W_dec = np.random.randn(3,2)*0.5

lr = 0.03
epochs = 1000
beta = 0.01

# =========================
# Funciones
# =========================

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(y):
    return y*(1-y)

# =========================
# Entrenamiento
# =========================

for epoch in range(epochs):

    # -------- Encoder --------
    
    mu = W_mu @ x
    logvar = W_logvar @ x
    
    sigma = np.exp(0.5*logvar)
    
    eps = np.random.randn(2)
    
    z = mu + sigma * eps
    
    # -------- Decoder --------
    
    h = W_dec @ z
    x_hat = sigmoid(h)
    
    # -------- Loss --------
    
    recon_loss = np.sum((x_hat-x)**2)
    
    kl_loss = -0.5*np.sum(
        1 + logvar - mu**2 - np.exp(logvar)
    )
    
    loss = recon_loss + beta*kl_loss
    
    # -------- Backprop decoder --------
    
    d_recon = 2*(x_hat-x)
    
    delta = d_recon * dsigmoid(x_hat)
    
    grad_W_dec = np.outer(delta,z)
    
    dz = W_dec.T @ delta
    
    # -------- Backprop encoder --------
    
    dmu = dz + beta*mu
    
    dlogvar = dz*eps*0.5*np.exp(0.5*logvar) \
              + beta*0.5*(np.exp(logvar)-1)
    
    grad_W_mu = np.outer(dmu,x)
    grad_W_logvar = np.outer(dlogvar,x)
    
    # -------- Update --------
    
    W_dec -= lr*grad_W_dec
    W_mu -= lr*grad_W_mu
    W_logvar -= lr*grad_W_logvar
    
    if epoch%20==0:
        print(
            "epoch:",epoch,
            "loss:",round(loss,4),
            "recon:",round(recon_loss,4),
            "kl:",round(kl_loss,4)
        )

print("\nEntrada:",x)
print("Reconstrucción:",x_hat)
