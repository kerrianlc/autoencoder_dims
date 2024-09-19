import torchvision.transforms as transforms
from torchvision import datasets
import torch 

class Loader:
    def __init__(self):
        tensor_transform = transforms.ToTensor()

        self.dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=tensor_transform
        )
        self.loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)