import math
import random

# -----------------------------
# Funciones auxiliares
# -----------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    # derivada usando salida ya activada
    return y * (1 - y)

# -----------------------------
# Inicialización de pesos
# -----------------------------

# Encoder (3 -> 2)
W = [
    [0.3, 0.2, 0.4],  # z1
    [0.1, 0.5, 0.2]   # z2
]

#W = [
#    [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],  # z1
#    [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]   # z2
#]


# Decoder (2 -> 3)
V = [
     [0.3, 0.4],  # x̂1
     [0.2, 0.5],  # x̂2
     [0.6, 0.1]   # x̂3
    ]

#V = [
#    [random.uniform(-1, 1), random.uniform(-1, 1)],  # x̂1
#    [random.uniform(-1, 1), random.uniform(-1, 1)],  # x̂2
#    [random.uniform(-1, 1), random.uniform(-1, 1)]   # x̂3
#]

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
        #z.append(sigmoid(suma))
 
  #  print(z)
    # Decoder
    x_hat = []
    for i in range(3):
        suma = 0
        for j in range(2):
            suma += V[i][j] * z[j]
        #x_hat.append(sigmoid(suma))
        x_hat.append(suma)
  #  print(x_hat)
    # ---- ERROR ----
    error = 0
    for i in range(3):
        error += (x_hat[i] - x[i]) ** 2
 #       error += (x[i] - x_hat[i]) ** 2

    # ---- BACKPROPAGATION ----

    # Error en salida
    delta_output = []
    for i in range(3):
        e = x_hat[i] - x[i]
 #       e = x[i] - x_hat[i]
 #       print('error', e)
        #delta_output.append(e * dsigmoid(x_hat[i]))
        delta_output.append((e * z[0]) + (e * z[1]))
        #delta_output.append(e * x_hat[i])

    #print('delta w', delta_output)

    # Error en capa latente
    delta_hidden = []
    for i in range(2):
        error_latente = 0
        for j in range(3):
            error_latente += (x_hat[j] - x[j]) * V[j][i]
            #error_latente += delta_output[j] * V[j][i]
        delta_hidden.append(error_latente)
        #delta_hidden.append(error_latente * z[i])
        #delta_hidden.append(error_latente * dsigmoid(z[i]))

    # ---- ACTUALIZACIÓN DE PESOS ----

    # Actualizar V (decoder)
    for i in range(3):
        for j in range(2):
            #print(V[i][j], delta_output[i])
            V[i][j] -= learning_rate * (z[j] * ( x_hat[i] - x[i])  ) 
            #V[i][j] -= learning_rate * delta_output[i] * z[j]
    # print(V)

    # Actualizar W (encoder)
    for i in range(2):
        for j in range(3):
            #print(W[i][j], delta_hidden[i], learning_rate, x[j])
            W[i][j] = W[i][j] - learning_rate * delta_hidden[i] * x[j]
            #W[i][j] -= learning_rate * delta_hidden[i] * x[j]
 #   print(W)
 #   print(V)
    print(x_hat, error)
    # Mostrar progreso
#    if epoch % 1000 == 0:
 #   print("Epoch:", epoch, "Error:", error)

# -----------------------------
# Resultado final
# -----------------------------
print("\nEntrada original:", x)
print("Reconstrucción:", x_hat)

