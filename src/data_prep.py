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
        augmentations_: list = [],
) -> DataLoader:
    augmentations = (augmentations_ if use_augmentations else [])
    transform = transforms.Compose([*augmentations,
                                    #transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    #transforms.Normalize(mean=CINIC_MEAN,std=CINIC_STD)
                                    ])

    ds = torchvision.datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return loader

def get_cinic(
        data_path: str,
        batch_size: int = 32
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")
    test_path = os.path.join(data_path, "test")

    cinic_train = get_dataset(train_path, batch_size, True, True, augmentations_train_aug_5)
    cinic_validation = get_dataset(valid_path, batch_size, True, True, augmentations_val_test_aug_5)
    cinic_test = get_dataset(test_path, batch_size, True, True, augmentations_val_test_aug_5)

    return cinic_train, cinic_validation, cinic_test

def apply_cutout(img):
    return cutout_transform(img, n_holes=1, length=50)

def cutout_transform(img: torch.Tensor, n_holes: int = 1, length: int = 50) -> torch.Tensor:
    """
    Apply the Cutout augmentation to an image tensor.

    Arguments:
    img (torch.Tensor): Image tensor of size (C, H, W).
    n_holes (int): Number of patches to cut out from the image.
    length (int): Side length of each square patch.
    
    Result:
    torch.Tensor: The image tensor after applying the cutout augmentation.
    """
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)

    for _ in range(n_holes):
        # Randomly choose a center for the hole
        y = np.random.randint(h)
        x = np.random.randint(w)

        # Determine the coordinates of the patch
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.

    mask = torch.from_numpy(mask).expand_as(img)
    return img * mask

augmentations_train_EfficientNet = [
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),    
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                           saturation=0.2, hue=0.1),
]

augmentations_val_test_EfficientNet = [
    transforms.Resize(224),
]

### Augmentations exploring

# No augmentations
augmentations_train_aug_1 = [
    transforms.Resize(224),
]

augmentations_val_test_aug_1 = [
    transforms.Resize(224),
]

augmentations_train_aug_2 = [
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20), 
]

augmentations_val_test_aug_2 = [
    transforms.Resize(224),
]

augmentations_train_aug_3 = [
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                           saturation=0.2, hue=0.1),
]

augmentations_val_test_aug_3 = [
    transforms.Resize(224),
]

augmentations_train_aug_4 = [
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(apply_cutout),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

augmentations_val_test_aug_4 = [
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

augmentations_train_aug_5 = [
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Lambda(apply_cutout),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

augmentations_val_test_aug_5 = [
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]