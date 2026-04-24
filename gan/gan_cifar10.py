import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Configurar random seeds para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# ===== PARÁMETROS =====
NOISE_DIM = 100
IMG_SHAPE = (32, 32, 3)  # CIFAR-10: 32x32 color
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.0002

# ===== GENERADOR =====
def build_generator():
    """
    Crea el generador que convierte ruido en imágenes CIFAR-10.
    Entrada: vector de ruido (100,)
    Salida: imagen (32, 32, 3)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(NOISE_DIM,)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(int(np.prod(IMG_SHAPE)), activation='tanh'),
        tf.keras.layers.Reshape(IMG_SHAPE)
    ])
    return model

# ===== DISCRIMINADOR =====
def build_discriminator():
    """
    Crea el discriminador que clasifica si una imagen es real o fake.
    Entrada: imagen (32, 32, 3)
    Salida: probabilidad (0=fake, 1=real)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=IMG_SHAPE),
        
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# ===== CREAR MODELOS =====
generator = build_generator()
discriminator = build_discriminator()

# ===== COMPILAR MODELOS =====
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5)

cross_entropy = tf.keras.losses.BinaryCrossentropy()

# ===== FUNCIONES DE LOSS =====
def generator_loss(fake_output):
    """El generador quiere que el discriminador piense que las imágenes falsas son reales"""
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    """El discriminador quiere clasificar correctamente imágenes reales y falsas"""
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# ===== PASO DE ENTRENAMIENTO =====
@tf.function
def train_step(real_images):
    # Generar ruido
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    
    # ===== ENTRENAR GENERADOR =====
    with tf.GradientTape() as gen_tape:
        fake_images = generator(noise, training=True)
        fake_output = discriminator(fake_images, training=True)
        gen_loss = generator_loss(fake_output)
    
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    
    # ===== ENTRENAR DISCRIMINADOR =====
    with tf.GradientTape() as disc_tape:
        real_output = discriminator(real_images, training=True)
        fake_images = generator(noise, training=True)
        fake_output = discriminator(fake_images, training=True)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# ===== CARGAR Y PREPARAR DATOS =====
print("Cargando dataset CIFAR-10...")
(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()

# Normalizar imágenes a rango [-1, 1]
x_train = (x_train.astype('float32') - 127.5) / 127.5

# Crear dataset
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(BATCH_SIZE)

# ===== ENTRENAR GAN =====
print("Iniciando entrenamiento con CIFAR-10...")
losses_g = []
losses_d = []

for epoch in range(EPOCHS):
    gen_loss_sum = 0
    disc_loss_sum = 0
    num_batches = 0
    
    for real_images in train_dataset:
        gen_loss, disc_loss = train_step(real_images)
        gen_loss_sum += gen_loss
        disc_loss_sum += disc_loss
        num_batches += 1
    
    avg_gen_loss = gen_loss_sum / num_batches
    avg_disc_loss = disc_loss_sum / num_batches
    losses_g.append(avg_gen_loss)
    losses_d.append(avg_disc_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} - Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")

print("¡Entrenamiento completado!")

# ===== GENERAR IMÁGENES =====
def generate_and_show_images(num_images=16):
    noise = tf.random.normal([num_images, NOISE_DIM])
    generated_images = generator(noise, training=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, img in zip(axes.flat, generated_images):
        # Desnormalizar imagen
        img = (img.numpy() * 127.5 + 127.5).astype('uint8')
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('generated_cifar10.png', dpi=150)
    print("Imágenes generadas guardadas en 'generated_cifar10.png'")
    plt.show()

# ===== GRAFICAR LOSSES =====
plt.figure(figsize=(10, 5))
plt.plot(losses_g, label='Generator Loss')
plt.plot(losses_d, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('GAN Training - CIFAR-10')
plt.savefig('training_losses_cifar10.png', dpi=150)
print("Gráfico de pérdidas guardado en 'training_losses_cifar10.png'")
plt.show()

# Generar imágenes
generate_and_show_images()
