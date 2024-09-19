
from .transform_factory import TRANSFORMS
from torchvision import datasets
import torch


class Loader:
    def __init__(self, batch=64, transform=""):
        tensor_transform = TRANSFORMS[transform]
        tensor_transform = tensor_transform.transforms

        self.dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=tensor_transform
        )
        self.loader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=batch, shuffle=True
        )

