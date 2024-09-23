import matplotlib.pyplot as plt
import torch
from dataload.data_loader import Loader


def plot_loss(losses):
    loss_l = losses
    if type(losses) != list:
        loss_l = [losses]
    plt.title("Training loss")
    for loss in loss_l:
        plt.plot([tensor.detach().numpy() for tensor in loss])
    plt.show()


def plot_reconstruction(model):
    loader_inst = Loader(transform="", batch=1)
    loader = loader_inst.loader
    model.eval()
    for image, _ in loader:
        item = image.reshape(-1, 28 * 28)
        print(item.shape, image.shape)
        plt.imshow(image[0, 0])
        plt.show()

        reco = model.encoder(item).detach().numpy()
        reco = model.forward(item).detach().numpy()

        plt.imshow(torch.Tensor(reco.reshape(-1, 28, 28)[0]) - image[0, 0])
        plt.show()