import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision import transforms



CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC_STD = [0.24205776, 0.23828046, 0.25874835]


def get_dataset_ensemble(
        path: str,
        batch_size: int,
        shuffle: bool,
        use_augmentations: bool,
        augmentations_: list = [],
) -> DataLoader:
    augmentations = (augmentations_ if use_augmentations else [])
    transform = transforms.Compose([*augmentations,
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    ds = torchvision.datasets.ImageFolder(path, transform=transform
                                            , target_transform=binary_target_transform
                                            )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    ds = loader.dataset
    ds.targets = [binary_mapping[label] for label in ds.targets]
    ds.classes = ['animal', 'vehicle']
    ds.class_to_idx = {'animal': 0, 'vehicle': 1}
    return loader

def get_subset_ensemble(
        path: str,
        subset: str,  # "animals" or "vehicles"
        batch_size: int,
        shuffle: bool,
        use_augmentations: bool,
        augmentations_: list = []
) -> DataLoader:
    if subset == "animals":
        target_transform = animal_target_transform
        allowed_labels = [2, 3, 4, 5, 6, 7]
        new_classes = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
    elif subset == "vehicles":
        target_transform = vehicle_target_transform
        allowed_labels = [0, 1, 8, 9]
        new_classes = ['airplane', 'automobile', 'ship', 'truck']
    else:
        raise ValueError("subset must be either 'animals' or 'vehicles'")
    
    augmentations = (augmentations_ if use_augmentations else [])
    transform = transforms.Compose([*augmentations,
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    ds = torchvision.datasets.ImageFolder(path, transform=transform, target_transform=target_transform)

    if allowed_labels is not None:
        # ds.samples is a list of (filepath, label) tuples.
        ds.samples = [s for s in ds.samples if s[1] in allowed_labels]
        # Update targets accordingly.
        ds.targets = [s[1] for s in ds.samples]

    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    
    # Update metadata: even though the target_transform already remaps the labels on the fly,
    # we update the dataset metadata so that downstream code (e.g. evaluation) sees the correct classes.
    ds = loader.dataset
    ds.classes = new_classes
    ds.class_to_idx = {cls: i for i, cls in enumerate(new_classes)}
    return loader

def get_cinic_ensemble(
        data_path: str,
        batch_size: int = 32
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")
    test_path = os.path.join(data_path, "test")

    # Binary
    cinic_train = get_dataset_ensemble(train_path, batch_size, True, True, augmentations_train_EfficientNet)
    cinic_validation = get_dataset_ensemble(valid_path, batch_size, True, True, augmentations_val_test_EfficientNet)
    cinic_test = get_dataset_ensemble(test_path, batch_size, True, True, augmentations_val_test_EfficientNet)

    # Animals subset
    # cinic_train = get_subset_ensemble(train_path, "animals", batch_size, True, True, augmentations_train_EfficientNet)
    # cinic_validation = get_subset_ensemble(valid_path, "animals", batch_size, True, True, augmentations_val_test_EfficientNet)
    # cinic_test = get_subset_ensemble(test_path, "animals", batch_size, True, True, augmentations_val_test_EfficientNet)

    # Vehicles subset
    # cinic_train = get_subset_ensemble(train_path, "vehicles", batch_size, True, True, augmentations_train_EfficientNet)
    # cinic_validation = get_subset_ensemble(valid_path, "vehicles", batch_size, True, True, augmentations_val_test_EfficientNet)
    # cinic_test = get_subset_ensemble(test_path, "vehicles", batch_size, True, True, augmentations_val_test_EfficientNet)


    return cinic_train, cinic_validation, cinic_test

# Mapping for converting the original 10-class CINIC10 labels into binary labels:
# vehicles (0: airplane, 1: automobile, 8: ship, 9: truck) as 1
# and animals (2: bird, 3: cat, 4: deer, 5: dog, 6: frog, 7: horse) as 0.
binary_mapping = {0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1}

def binary_target_transform(label):
    return binary_mapping[label]

def animal_target_transform(label):
    """Map original labels for animals (2,3,4,5,6,7) to new contiguous indices 0-5."""
    mapping = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    if label in mapping:
        return mapping[label]
    else:
        raise ValueError(f"Label {label} is not an animal.")

def vehicle_target_transform(label):
    """Map original labels for vehicles (0,1,8,9) to new contiguous indices 0-3."""
    mapping = {0: 0, 1: 1, 8: 2, 9: 3}
    if label in mapping:
        return mapping[label]
    else:
        raise ValueError(f"Label {label} is not a vehicle.")

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