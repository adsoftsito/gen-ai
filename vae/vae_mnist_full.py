import numpy as np
import gzip

def load_mnist_npz_gz(path):

    with gzip.open(path, 'rb') as f:
        data = np.load(f)

        x_train = data['x_train']
        x_test  = data['x_test']

        y_train = data['y_train']
        y_test  = data['y_test']

    # unir datos
    X = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    # normalizar
    X = X.astype(np.float32) / 255.0

    # flatten
    X = X.reshape(-1, 784)

    return X, y

# usar tu archivo
X, y = load_mnist_npz_gz("mnist.npz.gz")

# usar subset para rapidez
#X = X[:2000]
idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]


X = X[:2000]
y = y[:2000]

print(X.shape)

np.random.seed(0)

input_dim = 784
latent_dim = 2

W_mu = np.random.randn(latent_dim, input_dim)*0.01
W_logvar = np.random.randn(latent_dim, input_dim)*0.01
W_dec = np.random.randn(input_dim, latent_dim)*0.01

lr = 0.001
epochs = 500
batch_size = 32
beta = 0.001

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(y):
    return y*(1-y)

def bce(x_hat, x):
    return -np.sum(
        x*np.log(x_hat+1e-8) + (1-x)*np.log(1-x_hat+1e-8)
    )

for epoch in range(epochs):

    total_loss = 0

    # shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]

    for i in range(0, len(X), batch_size):

        batch = X[i:i+batch_size]

        # acumuladores
        gW_mu = np.zeros_like(W_mu)
        gW_logvar = np.zeros_like(W_logvar)
        gW_dec = np.zeros_like(W_dec)

        for x in batch:

            # -------- Encoder --------
            mu = W_mu @ x
            logvar = W_logvar @ x

            sigma = np.exp(0.5*logvar)
            eps = np.random.randn(latent_dim)
            z = mu + sigma * eps

            # -------- Decoder --------
            h = W_dec @ z
            x_hat = sigmoid(h)

            # -------- Loss --------
            recon = bce(x_hat, x)

            kl = -0.5*np.sum(
                1 + logvar - mu**2 - np.exp(logvar)
            )

            loss = recon + beta*kl
            total_loss += loss

            # -------- Backprop --------

            d_recon = (x_hat - x) / (x_hat*(1-x_hat)+1e-8)
            delta = d_recon * dsigmoid(x_hat)

            gW_dec += np.outer(delta, z)

            dz = W_dec.T @ delta

            dmu = dz + beta*mu

            dlogvar = dz*eps*0.5*np.exp(0.5*logvar) \
                      + beta*0.5*(np.exp(logvar)-1)

            gW_mu += np.outer(dmu, x)
            gW_logvar += np.outer(dlogvar, x)

        # update batch
        W_dec -= lr * gW_dec / batch_size
        W_mu -= lr * gW_mu / batch_size
        W_logvar -= lr * gW_logvar / batch_size

    print("epoch:", epoch, "loss:", round(total_loss,2))



import matplotlib.pyplot as plt

def show_reconstruction(i):
    x = X[i]

    mu = W_mu @ x
    z = mu  # sin ruido

    x_hat = sigmoid(W_dec @ z)

    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(x.reshape(28,28), cmap="gray")

    plt.subplot(1,2,2)
    plt.title("Reconstrucción")
    plt.imshow(x_hat.reshape(28,28), cmap="gray")

#    plt.show()
    plt.savefig("reconstruccion.png")
    plt.close()
show_reconstruction(3)


def generate(z1, z2):
    z = np.array([z1, z2])
    x_hat = sigmoid(W_dec @ z)

    plt.imshow(x_hat.reshape(28,28), cmap="gray")
    #plt.show()
    plt.savefig("generated.png")
    plt.close()

# ejemplo
#generate(1,1)
#generate(4,4)
#generate(-2,-2)

def generate_digit(digit):

    # buscar un ejemplo en el dataset
    for i in range(len(X)):
        if y[i] == digit:
            x = X[i]
            break

    # encoder
    mu = W_mu @ x
    z = mu   # sin ruido para reconstrucción

    # decoder
    x_hat = sigmoid(W_dec @ z)

    # mostrar
    import matplotlib.pyplot as plt

    plt.subplot(1,2,1)
    plt.title(f"Original {digit}")
    plt.imshow(x.reshape(28,28), cmap="gray")

    plt.subplot(1,2,2)
    plt.title("Reconstrucción")
    plt.imshow(x_hat.reshape(28,28), cmap="gray")

    plt.savefig(f"digit_{digit}.png")
    plt.close()

generate_digit(3)
#generate_digit(7)
#generate_digit(0)
