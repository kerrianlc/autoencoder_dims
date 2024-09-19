import matplotlib.pyplot as plt

from dataload.data_loader import Loader


def plot_loss(losses):
    plt.title("Training loss")
    plt.plot([tensor.detach().numpy() for tensor in losses])
    plt.show()


def plot_reconstruction(model):
    loader_inst = Loader(transform="", batch=1)
    loader = loader_inst.loader
    for image, _ in loader:
        item = image.reshape(-1, 28 * 28)
        print(item.shape, image.shape)
        plt.imshow(image[0, 0])
        plt.show()

        reco = model.encoder(item).detach().numpy()
        reco = model.forward(item).detach().numpy()

        plt.imshow(reco.reshape(-1, 28, 28)[0])
        plt.show()