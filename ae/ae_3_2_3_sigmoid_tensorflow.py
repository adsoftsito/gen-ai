import tensorflow as tf
import numpy as np

# -----------------------------
# Datos
# -----------------------------
x = np.array([[1,0,1]], dtype=np.float32)

# -----------------------------
# Modelo Autoencoder 3-2-3
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='sigmoid', input_shape=(3,)),  # Encoder
    tf.keras.layers.Dense(3, activation='sigmoid')                     # Decoder
])

# -----------------------------
# Compilación
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
    loss='mse'
)

# -----------------------------
# Entrenamiento
# -----------------------------
model.fit(x, x, epochs=200, verbose=1)

# -----------------------------
# Reconstrucción
# -----------------------------
x_hat = model.predict(x)

print("Entrada original:", x)
print("Reconstrucción:", x_hat)
