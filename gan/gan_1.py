import numpy as np

# Activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# Inicialización
np.random.seed(42)

# Generador
Wg = np.random.randn(2, 2)

# Discriminador
Wd = np.random.randn(2, 1)

lr = 0.01

# Datos reales (ejemplo simple)
real_data = np.array([[1,1],[0.9,0.8],[1.1,1.2]])

for epoch in range(1000):

    # ---------
    # Train Discriminator
    # ---------
    
    # Datos reales
    x_real = real_data[np.random.randint(0, len(real_data))]
    d_real = sigmoid(x_real @ Wd)

    # Datos falsos
    z = np.random.randn(2)
    x_fake = sigmoid(z @ Wg)
    d_fake = sigmoid(x_fake @ Wd)

    # Loss gradiente
    d_loss = -(np.log(d_real) + np.log(1 - d_fake))

    # Gradientes discriminador
    grad_d = (d_real - 1) * x_real.reshape(-1,1) + d_fake * x_fake.reshape(-1,1)

    Wd -= lr * grad_d

    # ---------
    # Train Generator
    # ---------
    
    z = np.random.randn(2)
    x_fake = sigmoid(z @ Wg)
    d_fake = sigmoid(x_fake @ Wd)

    # Gradiente generador
    grad_g = (d_fake - 1) * Wd.T * sigmoid_deriv(x_fake)

    Wg -= lr * np.outer(z, grad_g)

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | D(real): {d_real} | D(fake): {d_fake}")
