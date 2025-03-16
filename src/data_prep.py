import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision import transforms


CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC_STD = [0.24205776, 0.23828046, 0.25874835]


def get_dataset(
        path: str,
        batch_size: int,
        shuffle: bool,
        use_augmentations: bool,
) -> DataLoader:
    augmentations = ([
        # TODO: add augmentations
    ] if use_augmentations else [])
    transform = transforms.Compose([*augmentations,
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=CINIC_MEAN,std=CINIC_STD)])

    ds = torchvision.datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=3)
    return loader


def get_cinic(
        data_path: str,
        batch_size: int = 256
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")
    test_path = os.path.join(data_path, "test")

    cinic_train = get_dataset(train_path, batch_size, True, True)
    cinic_validation = get_dataset(valid_path, batch_size, True, False)
    cinic_test = get_dataset(test_path, batch_size, False, False)

    return cinic_train, cinic_validation, cinic_test
