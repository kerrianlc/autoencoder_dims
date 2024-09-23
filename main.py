from dataload.data_loader import Loader, PCALoader
from model import AutoEncoder, ReducedAutoEncoder
from plot import plot_loss, plot_reconstruction
import torch


model = AutoEncoder()
loader_inst = Loader(transform="")
pca_inst = PCALoader(transform="")
model = AutoEncoder()
pca_model = ReducedAutoEncoder()
mse_loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
train_loader = loader_inst.loader
pca_loader = pca_inst.loader
epochs = 5 # Constant loss after few epochs

outputs = []
losses, losses_pca = [], []
model.train()
pca_model.train()
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

print("_________MODEL TRAINING_________")
for epoch in range(epochs):
    for image, _ in pca_loader:
        reconstructed = pca_model(image)
        loss = mse_loss(reconstructed, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch}/{epochs}: loss={loss}")
        losses_pca.append(loss)
    outputs.append((epochs, image, reconstructed))

plot_loss([losses, losses_pca])
# plot_reconstruction(model)