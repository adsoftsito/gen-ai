import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# =========================
# 1. DESCARGA CELEBA
# =========================

import gdown

url = "https://drive.google.com/uc?id=1K7w3eY7z0t4Xx1zQwH0kC6q4ZlQ5ZxYB"  # puedes cambiar por mirror
output = "celeba.zip"
DATASET_AVAILABLE = True

if not os.path.exists("celeba"):
    try:
        print("Intentando descargar dataset CelebA...")
        gdown.download(url, output, quiet=False)
        
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall("celeba")
        print("Dataset descargado exitosamente.")
    except Exception as e:
        print(f"⚠️ Error al descargar dataset: {e}")
        print("Usando datos sintéticos para entrenar...")
        DATASET_AVAILABLE = False

# =========================
# 2. PARÁMETROS
# =========================

IMG_SIZE = 64
BATCH_SIZE = 64
LATENT_DIM = 100
EPOCHS = 50
N_CRITIC = 5
LAMBDA_GP = 10

# =========================
# 3. DATASET
# =========================

def load_images(path="celeba/img_align_celeba"):
    """Carga imágenes de CelebA o genera datos sintéticos si no están disponibles."""
    try:
        files = os.listdir(path)
        imgs = []

        for f in files[:20000]:  # limitar para pruebas
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(path, f), target_size=(IMG_SIZE, IMG_SIZE))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = (img / 127.5) - 1.0
            imgs.append(img)

        print(f"Dataset cargado: {len(imgs)} imágenes")
        return np.array(imgs)
    except Exception as e:
        print(f"⚠️ No se pudo cargar imágenes de {path}: {e}")
        print("Generando datos sintéticos para entrenaramiento...")
        # Generar imágenes aleatorias normalizadas (simulan rostros)
        synthetic_data = np.random.uniform(-1.0, 1.0, (5000, IMG_SIZE, IMG_SIZE, 3))
        return synthetic_data.astype(np.float32)

X_train = load_images()

# atributos dummy (ejemplo: smiling)
y_train = np.random.randint(0, 2, (len(X_train), 1))

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(10000).batch(BATCH_SIZE)

# =========================
# 4. GENERADOR
# =========================

def build_generator():
    z = Input(shape=(LATENT_DIM,))
    y = Input(shape=(1,))

    x = Concatenate()([z, y])

    x = Dense(4*4*1024)(x)
    x = Reshape((4,4,1024))(x)

    x = Conv2DTranspose(512, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    out = Conv2DTranspose(3, 4, strides=2, padding='same',
                          activation='tanh')(x)

    return Model([z, y], out)

# =========================
# 5. CRITIC (SIN SIGMOID)
# =========================

def build_critic():
    img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    y = Input(shape=(1,))

    y_map = Dense(IMG_SIZE*IMG_SIZE)(y)
    y_map = Reshape((IMG_SIZE, IMG_SIZE, 1))(y_map)

    x = Concatenate()([img, y_map])

    x = Conv2D(64, 4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    out = Dense(1)(x)

    return Model([img, y], out)

G = build_generator()
D = build_critic()

# =========================
# 6. OPTIMIZADORES
# =========================

opt_G = tf.keras.optimizers.Adam(0.0001, beta_1=0.0, beta_2=0.9)
opt_D = tf.keras.optimizers.Adam(0.0001, beta_1=0.0, beta_2=0.9)

# =========================
# 7. GRADIENT PENALTY
# =========================

def gradient_penalty(real, fake, y):
    alpha = tf.random.uniform([real.shape[0], 1,1,1], 0.0, 1.0)
    interpolated = alpha * real + (1 - alpha) * fake

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = D([interpolated, y])

    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,2,3]))

    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

# =========================
# 8. TRAIN STEP
# =========================

@tf.function
def train_step(real_imgs, y):

    for _ in range(N_CRITIC):

        z = tf.random.normal([BATCH_SIZE, LATENT_DIM])

        with tf.GradientTape() as tape:
            fake_imgs = G([z, y], training=True)

            d_real = D([real_imgs, y], training=True)
            d_fake = D([fake_imgs, y], training=True)

            gp = gradient_penalty(real_imgs, fake_imgs, y)

            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + LAMBDA_GP * gp

        grads = tape.gradient(d_loss, D.trainable_variables)
        opt_D.apply_gradients(zip(grads, D.trainable_variables))

    # ---- Generator ----
    z = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as tape:
        fake_imgs = G([z, y], training=True)
        d_fake = D([fake_imgs, y], training=True)

        g_loss = -tf.reduce_mean(d_fake)

    grads = tape.gradient(g_loss, G.trainable_variables)
    opt_G.apply_gradients(zip(grads, G.trainable_variables))

    return d_loss, g_loss

# =========================
# 9. TRAIN LOOP
# =========================

for epoch in range(EPOCHS):

    for real_imgs, y in dataset:
        d_loss, g_loss = train_step(real_imgs, y)

    print(f"Epoch {epoch} | D: {d_loss:.4f} | G: {g_loss:.4f}")

    if epoch % 5 == 0:
        z = tf.random.normal([5, LATENT_DIM])
        y_test = tf.ones((5,1))
        imgs = G([z, y_test])

        imgs = (imgs + 1) / 2.0

        for i in range(5):
            plt.imshow(imgs[i])
            plt.axis('off')
            plt.show()

# =========================
# 10. GUARDAR MODELOS
# =========================

G.save("generator_cwgan_gp.h5")
D.save("critic_cwgan_gp.h5")

print("Modelos guardados correctamente")