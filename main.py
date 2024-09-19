from dataload.data_loader import Loader
from model import AutoEncoder
import torch
import matplotlib.pyplot as plt

model = AutoEncoder()
loader_inst = Loader(transform="")
model = AutoEncoder()
mse_loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
train_loader = loader_inst.loader
epochs = 5 # Constant loss after few epochs

outputs = []
losses = []
model.train()
print("_________MODEL TRAINING_________")
for epoch in range(epochs):
    for image, _ in train_loader:
        image = image.reshape(-1, 28 * 28)
        reconstructed = model(image)
        loss = mse_loss(reconstructed, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch}/{epochs}: loss={loss}")
        losses.append(loss)
    outputs.append((epochs, image, reconstructed))


plt.plot([tensor.detach().numpy() for tensor in losses])
plt.show()