import math
import random

# -----------------------------
# Funciones
# -----------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

# -----------------------------
# Crear número 2 (8x5)
# -----------------------------

numero_2 = [
    0,1,1,1,0,
    1,0,0,0,1,
    0,0,0,0,1,
    0,0,0,1,0,
    0,0,1,0,0,
    0,1,0,0,0,
    1,0,0,0,0,
    1,1,1,1,1
]

x = numero_2[:]  # copia

INPUT = 40
HIDDEN = 10
OUTPUT = 40

learning_rate = 0.5

# -----------------------------
# Inicializar pesos
# -----------------------------

# Encoder 40 -> 10
W = []
for i in range(HIDDEN):
    fila = []
    for j in range(INPUT):
        fila.append(random.uniform(-1,1))
    W.append(fila)

# Decoder 10 -> 40
V = []
for i in range(OUTPUT):
    fila = []
    for j in range(HIDDEN):
        fila.append(random.uniform(-1,1))
    V.append(fila)

# -----------------------------
# Entrenamiento
# -----------------------------

for epoch in range(5000):

    # ---- FORWARD ----
    
    # Encoder
    z = []
    for i in range(HIDDEN):
        suma = 0
        for j in range(INPUT):
            suma += W[i][j] * x[j]
        z.append(sigmoid(suma))

    # Decoder
    x_hat = []
    for i in range(OUTPUT):
        suma = 0
        for j in range(HIDDEN):
            suma += V[i][j] * z[j]
        x_hat.append(sigmoid(suma))

    # ---- ERROR ----
    error = 0
    for i in range(OUTPUT):
        error += (x[i] - x_hat[i]) ** 2

    # ---- BACKPROP ----

    # Output delta
    delta_output = []
    for i in range(OUTPUT):
        e = x[i] - x_hat[i]
        delta_output.append(e * dsigmoid(x_hat[i]))

    # Hidden delta
    delta_hidden = []
    for i in range(HIDDEN):
        error_latente = 0
        for j in range(OUTPUT):
            error_latente += delta_output[j] * V[j][i]
        delta_hidden.append(error_latente * dsigmoid(z[i]))

    # ---- Actualizar V ----
    for i in range(OUTPUT):
        for j in range(HIDDEN):
            V[i][j] += learning_rate * delta_output[i] * z[j]

    # ---- Actualizar W ----
    for i in range(HIDDEN):
        for j in range(INPUT):
            W[i][j] += learning_rate * delta_hidden[i] * x[j]

    if epoch % 500 == 0:
        print("Epoch:", epoch, "Error:", error)

# -----------------------------
# Mostrar reconstrucción
# -----------------------------

print("\nReconstrucción final:\n")

for i in range(8):
    fila = ""
    for j in range(5):
        valor = x_hat[i*5 + j]
        if valor > 0.5:
            fila += "1 "
        else:
            fila += "0 "
    print(fila)

