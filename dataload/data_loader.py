
from .transform_factory import TRANSFORMS
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.decomposition import PCA


class Loader:
    def __init__(self, batch=64, transform=""):
        tensor_transform = TRANSFORMS[transform]
        tensor_transform = tensor_transform.transforms

        self.dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=tensor_transform
        )
        self.loader = DataLoader(
            dataset=self.dataset, batch_size=batch, shuffle=True
        )

class PCADataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class PCALoader:
    def __init__(self, batch=64, transform=""):
        tensor_transform = TRANSFORMS[transform]
        tensor_transform = tensor_transform.transforms

        self.dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=tensor_transform
        )
        data_list, labels_list = [], []
        for input, label in DataLoader(
            dataset=self.dataset, batch_size=len(self.dataset)
        ):
            data_list.append(input)
            labels_list.append(label)
        input = torch.cat(data_list, dim=0)
        input = input.reshape(-1, 28 * 28)
        label = torch.cat(labels_list, dim=0)
        self.pca = PCA(n_components=512)
        components = torch.tensor(self.pca.fit_transform(input), dtype=torch.float32)

        self.pca_dataset = PCADataset(components, label)
        self.loader = DataLoader(
            dataset=self.pca_dataset, batch_size=batch, shuffle=True
        )