import math
import random

# -----------------------------
# Inicialización de pesos
# -----------------------------

# Encoder (3 -> 2)
W = [
    [0.3, 0.2, 0.4],  # z1
    [0.1, 0.5, 0.2]   # z2
]


# Decoder (2 -> 3)
V = [
     [0.3, 0.4],  # x̂1
     [0.2, 0.5],  # x̂2
     [0.6, 0.1]   # x̂3
    ]


learning_rate = 0.1

# -----------------------------
# Dato de entrenamiento
# -----------------------------
x = [1, 0, 1]   # ejemplo

# -----------------------------
# Entrenamiento
# -----------------------------
for epoch in range(20):

    # ---- FORWARD PASS ----

    # Encoder
    z = []
    for i in range(2):
        suma = 0
        for j in range(3):
            suma += W[i][j] * x[j]
        z.append(suma)
 
  #  print(z)
    # Decoder
    x_hat = []
    for i in range(3):
        suma = 0
        for j in range(2):
            suma += V[i][j] * z[j]
        x_hat.append(suma)
  #  print(x_hat)

  # ---- ERROR ----
    error = 0
    for i in range(3):
        error += (x_hat[i] - x[i]) ** 2

    # ---- BACKPROPAGATION ----

    # Error en salida
    delta_output = []
    for i in range(3):
        e = x_hat[i] - x[i]
        delta_output.append((e * z[0]) + (e * z[1]))


    # Error en capa latente
    delta_hidden = []
    for i in range(2):
        error_latente = 0
        for j in range(3):
            error_latente += (x_hat[j] - x[j]) * V[j][i]
        delta_hidden.append(error_latente)

    # ---- ACTUALIZACIÓN DE PESOS ----

    # Actualizar V (decoder)
    for i in range(3):
        for j in range(2):
            V[i][j] -= learning_rate * (z[j] * ( x_hat[i] - x[i])  ) 

    # Actualizar W (encoder)
    for i in range(2):
        for j in range(3):
            W[i][j] = W[i][j] - learning_rate * delta_hidden[i] * x[j]
    print(x_hat, error)
    # Mostrar progreso

# -----------------------------
# Resultado final
# -----------------------------
print("\nEntrada original:", x)
print("Reconstrucción:", x_hat)

