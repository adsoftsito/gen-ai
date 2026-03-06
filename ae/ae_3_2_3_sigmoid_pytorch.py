import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Datos
# -----------------------------
x = torch.tensor([[1.,0.,1.]])

# -----------------------------
# Modelo Autoencoder
# -----------------------------
class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(3,2),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(2,3),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

model = Autoencoder()

# -----------------------------
# Entrenamiento
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

for epoch in range(200):

    optimizer.zero_grad()

    output = model(x)

    loss = criterion(output, x)

    loss.backward()

    optimizer.step()

    print("Epoch:", epoch, "Error:", loss.item())

# -----------------------------
# Resultado
# -----------------------------
print("\nEntrada original:", x)
print("Reconstrucción:", model(x))
