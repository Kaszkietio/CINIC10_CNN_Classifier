import os
import argparse
import json
import torch
from torch import nn

from data_prep import get_cinic
from utils import set_seed, get_device
from custom_models import ResNet_32x32, AlexNet_32x32


MODELS = {
    "alexnet": AlexNet_32x32,
    "resnet": ResNet_32x32
}


OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.AdamW
}


def train(config_path: str, data_path: str):
    config = None
    with open(config_path) as f:
        config_str = f.read()
        config = json.loads(config_str)

    # Set seed for reproducibility
    seed = int(config["seed"]) if "seed" in config else 0
    set_seed(seed)

    device = get_device()

    cinic_train, cinic_valid, cinic_test = get_cinic(data_path, batch_size=256)
    model: nn.Module = MODELS[config["model"]](**config["model_params"]).to(device)
    optimizer: torch.optim.Optimizer = OPTIMIZERS[config["optimizer"]](params=model.parameters(),
                                                **config["optimizer_params"])
    loss_fn = nn.CrossEntropyLoss()

    epochs = int(config["epochs"])
    # Training
    for i in range(epochs):
        for input, target in cinic_train:
            optimizer.zero_grad()
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()


def setup_checkpoint_folder():
    pass


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    train(args.config, args.data_path)