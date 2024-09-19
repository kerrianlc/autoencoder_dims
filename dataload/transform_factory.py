import torchvision.transforms as transforms
from collections import defaultdict



class RandomTransform:
    def __init__(self):
        self.transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ToTensor(),
            ]
        )

class BasicTransform:
    def __init__(self):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


TRANSFORMS = defaultdict(BasicTransform)
TRANSFORMS["random"] =  RandomTransform