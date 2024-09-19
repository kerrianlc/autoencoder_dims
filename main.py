from data_loader import Loader
from model import AutoEncoder
import torch

model = AutoEncoder()
loader = Loader()
model = AutoEncoder()
mse_loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
epochs = 20

outputs = []
losses = []
model.train()
for epoch in range(epochs):
    for image, _ in loader:
        print(image)
        image = image.reshape(-1, 28 * 28)
        reconstructed = model(image)
        loss = mse_loss(reconstructed, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)
    outputs.append((epochs, image, reconstructed))
