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
        use_augmentations: bool,
) -> DataLoader:
    augmentations = ([
        # TODO: add augmentations
    ] if use_augmentations else [])
    transform = transforms.Compose([*augmentations,
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=CINIC_MEAN,std=CINIC_STD)])

    ds = torchvision.datasets.ImageFolder(path, transform=transform)
    return ds


def get_cinic(
        data_path: str,
        validation_size: float = 0.2,
        batch_size: int = 256
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    cinic_train_val = get_dataset(train_path, batch_size)
    cinic_len = len(cinic_train_val)
    idxs = np.arange(cinic_len)
    np.random.shuffle(idxs)

    validation_ds_size = int(cinic_len * validation_size)
    validation_idxs, train_idxs = np.split(idxs, [validation_ds_size])

    cinic_train = Subset(cinic_train_val, train_idxs)
    cinic_validation = Subset(cinic_train_val, validation_idxs)
    cinic_test = get_dataset(test_path, batch_size)

    return cinic_train, cinic_validation, cinic_test
