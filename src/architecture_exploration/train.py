import os
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from tqdm import tqdm

from custom_models import ResNet_32x32, AlexNet_32x32
from data_prep import get_cinic
from utils import set_seed, get_device


MODELS = {
    "AlexNet": AlexNet_32x32,
    "ResNet": ResNet_32x32
}

OPTIMIZERS = {
    "SGD": SGD,
    "AdamW": AdamW
}

SCHEDULERS = {
    "ExponentialLR": ExponentialLR
}


def train(config: dict, data_path: str):
    print("Starting training:")

    # Set seed for reproducibility
    seed = int(config["seed"]) if "seed" in config else 0
    set_seed(seed)
    print("Setting seed:", seed)


    device = get_device()
    print("Device: ", device)

    checkpoint = config["checkpoint_folder"]
    checkpoint = os.path.abspath(checkpoint)
    os.makedirs(checkpoint, exist_ok=True)
    print("Checkpoint folder:", checkpoint)

    with open(os.path.join(checkpoint, "config.json"), "w") as f:
        f.write(json.dumps(config))

    batch_size = int(config["batch_size"]) if "batch_size" in config else 256
    print("Batch size:", batch_size)
    cinic_train, cinic_valid, cinic_test = get_cinic(data_path, batch_size=batch_size)
    model: nn.Module = MODELS[config["model"]](**config["model_params"]).to(device)
    print("Model:", config["model"])
    print("Model params:", config["model_params"])
    optimizer: torch.optim.Optimizer = OPTIMIZERS[config["optimizer"]](params=model.parameters(),
                                                **config["optimizer_params"])
    print("Optimizer:", optimizer)
    scheduler = (
        SCHEDULERS[config["scheduler"]](optimizer, **config["scheduler_params"])
        if "scheduler" in config else None
    )
    print("Scheduler:", scheduler)
    criterion = nn.CrossEntropyLoss()

    epochs = int(config["epochs"])
    print("Number of epochs: ", epochs)
    metrics = {"loss": [float("+inf")], "accuracy": [0.0], "val_loss": [float("+inf")], "val_accuracy": [0.0]}
    best_model = model
    warmup_epochs = int(config["warmup_epochs"])
    print("Warmup epochs:", warmup_epochs)

    best_loss = float('inf')
    best_loss_epoch = -1

    classes = cinic_test.dataset.classes
    classes = sorted(classes, key=lambda x: cinic_test.dataset.class_to_idx[x])

    # Training
    for epoch in range(epochs):
        print("Epoch", epoch)

        print("Processing training")
        model.train()
        accuracy, loss = train_epoch(model, cinic_train, optimizer, criterion)
        metrics["accuracy"].append(accuracy)
        metrics["loss"].append(loss)

        print("Processing validation")
        model.eval()
        val_accuracy, val_loss = valid_epoch(model, cinic_valid, criterion)
        metrics["val_accuracy"].append(val_accuracy)
        metrics["val_loss"].append(val_loss)

        print(f"Loss: {metrics['loss'][-1]:.4f}", end=' ')
        print(f"Accuracy: {metrics['accuracy'][-1]:.4f}", end=' ')
        print(f"Validation Loss: {metrics['val_loss'][-1]:.4f}", end=' ')
        print(f"Validation Accuracy: {metrics['val_accuracy'][-1]:.4f}")
        print()

        # Saving checkpoint
        if epoch >= warmup_epochs and config["early_stopping"]["min_delta"] < best_loss - val_loss:
            best_model = model
            torch.save({
                    "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": criterion.state_dict(),
                    "epoch": epoch
            }, os.path.join(checkpoint, "state.pth"))

        # Early stopping
        if "early_stopping" in config:
            if config["early_stopping"]["min_delta"] < best_loss - val_loss:
                best_loss_epoch = epoch
                best_loss = val_loss
            elif epoch - best_loss_epoch >= config["early_stopping"]["patience"]:
                print("Early stopping!")
                break

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

    # Calculate test metrics and confusion matrix
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(os.path.join(checkpoint, "metrics.csv"))
    save_results(checkpoint, best_model, cinic_test, criterion, classes)


def train_epoch(
    model: nn.Module,
    train_ds: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss
):
    losses = []
    accuracies = []
    batch_sizes = []
    for input, target in tqdm(train_ds):
        optimizer.zero_grad()

        input, target = input.cuda(), target.cuda()
        ohe_target = F.one_hot(target, 10).type(torch.float32)
        output = model(input)
        loss = criterion(output, ohe_target)

        loss.backward()
        optimizer.step()

        pred = torch.argmax(output, dim=-1)
        accurate_pred = (pred == target).type(torch.float32)
        accuracies.append(torch.mean(accurate_pred).item())

        losses.append(loss.item())
        batch_sizes.append(len(input))

    accuracy = np.average(accuracies, weights=batch_sizes)
    loss = np.average(losses, weights=batch_sizes)
    return accuracy, loss


def valid_epoch(
    model: nn.Module,
    valid_ds: DataLoader,
    criterion: nn.CrossEntropyLoss
):
    val_losses = []
    val_accuracies = []
    batch_sizes = []
    with torch.no_grad():
        for input, target in tqdm(valid_ds):
            input, target = input.cuda(), target.cuda()
            ohe_target = F.one_hot(target, 10).type(torch.float32)
            output = model(input)
            val_loss = criterion(output, ohe_target)

            # Calculate accuracy
            pred = torch.argmax(output, dim=-1)
            accurate_pred = (pred == target).type(torch.float32)
            val_accuracies.append(torch.mean(accurate_pred).item())

            val_losses.append(val_loss.item())
            # Store information regarding
            batch_sizes.append(len(input))

    val_accuracy = np.average(val_accuracies, weights=batch_sizes)
    val_loss = np.average(val_losses, weights=batch_sizes)
    return val_accuracy, val_loss


def test_epoch(
    model: nn.Module,
    valid_ds: DataLoader,
    criterion: nn.CrossEntropyLoss
):
    val_losses = []
    val_accuracies = []
    batch_sizes = []
    predictions = []
    targets = []

    with torch.no_grad():
        for input, target in tqdm(valid_ds):
            input, target = input.cuda(), target.cuda()
            ohe_target = F.one_hot(target, 10).type(torch.float32)
            output = model(input)
            val_loss = criterion(output, ohe_target)

            # Calculate accuracy
            pred = torch.argmax(output, dim=-1)
            accurate_pred = (pred == target).type(torch.float32)
            val_accuracies.append(torch.mean(accurate_pred).item())

            val_losses.append(val_loss.item())
            batch_sizes.append(len(input))

            predictions.append(pred)
            targets.append(target)

    val_accuracy = np.average(val_accuracies, weights=batch_sizes)
    val_loss = np.average(val_losses, weights=batch_sizes)
    predictions = torch.concatenate(predictions).cpu().numpy()
    targets = torch.concatenate(targets).cpu().numpy()
    return val_accuracy, val_loss, targets, predictions


def save_results(
        checkpoint: str,
        model: nn.Module,
        cinic_test: DataLoader,
        criterion: nn.CrossEntropyLoss,
        classes: list[str]
):
    model.eval()
    test_accuracy, test_loss, test_targets, test_predictions = test_epoch(model,
                                                                          cinic_test, criterion)
    with open(os.path.join(checkpoint, "test_metrics.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.4f}")

    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(test_targets, test_predictions,
                                                   display_labels=classes)
    disp.figure_.savefig(os.path.join(checkpoint, "confusion_matrix_test.jpg"))

    # Confusion matrix without diagonal
    wrong_predictions_idx = test_targets != test_predictions
    test_targets = test_targets[wrong_predictions_idx]
    test_predictions = test_predictions[wrong_predictions_idx]
    disp = ConfusionMatrixDisplay.from_predictions(test_targets, test_predictions,
                                                   display_labels=classes)
    disp.figure_.savefig(os.path.join(checkpoint, "confusion_matrix_test_no_diag.jpg"))


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    config = None
    with open(args.config) as f:
        config_str = f.read()
        config = json.loads(config_str)
    train(config, args.data_path)