import os
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from data_prep_ensemble import get_cinic_ensemble
from utils import set_seed, get_device
from models_ensemble import EfficientNet_Transfer_Learning


MODELS = {
    "EfficientNet_Transfer_v2": EfficientNet_Transfer_Learning
    }

OPTIMIZERS = {
    "AdamW": torch.optim.AdamW
}

def train_ensemble_classifier(config_path: str, data_path: str):
    config = None
    with open(config_path) as f:
        config_str = f.read()
        config = json.loads(config_str)

    # Set seed for reproducibility
    seed = int(config["seed"]) if "seed" in config else 0
    set_seed(seed)

    device = get_device()
    print("Device: ", device)

    checkpoint = config["checkpoint_folder"]
    os.makedirs(checkpoint, exist_ok=True)

    with open(os.path.join(checkpoint, "config.json"), "w") as f:
        f.write(json.dumps(config))

    batch_size = int(config["batch_size"]) if "batch_size" in config else 256
    cinic_train, cinic_valid, cinic_test = get_cinic_ensemble(data_path, batch_size=batch_size)

    model: nn.Module = MODELS[config["model"]](**config["model_params"]).to(device)
    optimizer: torch.optim.Optimizer = OPTIMIZERS[config["optimizer"]](params=model.parameters(),
                                                **config["optimizer_params"])
    criterion = nn.CrossEntropyLoss()

    epochs = int(config["epochs"])
    metrics = {"loss": [float("+inf")], "accuracy": [0.0], "val_loss": [float("+inf")], "val_accuracy": [0.0]}
    best_model = model
    warmup_epochs = int(config["warmup_epochs"])

    best_loss = float('inf')
    best_loss_epoch = -1

    classes = cinic_test.dataset.classes
    classes = sorted(classes, key=lambda x: cinic_test.dataset.class_to_idx[x])

    # Training
    for epoch in range(epochs):
        print("Epoch", epoch)

        print("Processing training")
        accuracy, loss = train_epoch_ensemble(model, cinic_train, optimizer, criterion)
        metrics["accuracy"].append(accuracy)
        metrics["loss"].append(loss)

        print("Processing validation")
        val_accuracy, val_loss = valid_epoch_ensemble(model, cinic_valid, criterion)
        metrics["val_accuracy"].append(val_accuracy)
        metrics["val_loss"].append(val_loss)

        print(f"Loss: {metrics['loss'][-1]:.4f}", end=' ')
        print(f"Accuracy: {metrics['accuracy'][-1]:.4f}", end=' ')
        print(f"Validation Loss: {metrics['val_loss'][-1]:.4f}", end=' ')
        print(f"Validation Accuracy: {metrics['val_accuracy'][-1]:.4f}")

        # Saving checkpoint
        if epoch >= warmup_epochs and val_loss < metrics["val_loss"][-2]:
            best_model = model
            torch.save({
                    "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": criterion.state_dict(),
                    "epoch": epoch
            }, os.path.join(checkpoint, "state.pth"))

        # Early stopping
        if "early_stopping" in config:
            if val_loss < best_loss:
                best_loss_epoch = epoch
                best_loss = val_loss
            elif epoch - best_loss_epoch >= config["early_stopping"]["patience"]:
                print("Early stopping!")
                break

    # Calculate test metrics and confusion matrix
    test_accuracy, test_loss, test_targets, test_predictions = test_epoch_ensemble(best_model,
                                                                          cinic_test, criterion)
    with open(os.path.join(checkpoint, "test_metrics.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.4f}")

    # Save Scikit-learn report
    report = classification_report(test_targets, test_predictions, target_names=classes)
    with open(os.path.join(checkpoint, "classification_report.txt"), "w") as f:
        f.write(report)

    print(report)

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

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(checkpoint, "metrics.csv"))


def train_epoch_ensemble(
    model: nn.Module,
    train_ds: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss
):
    device = get_device()

    losses = []
    accuracies = []
    batch_sizes = []
    for input, target in tqdm(train_ds):
        optimizer.zero_grad()

        input, target = input.to(device), target.to(device)
        output = model(input)
        loss = criterion(output, target)

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


def valid_epoch_ensemble(
    model: nn.Module,
    valid_ds: DataLoader,
    criterion: nn.CrossEntropyLoss
):
    device = get_device()

    val_losses = []
    val_accuracies = []
    batch_sizes = []
    with torch.no_grad():
        for input, target in tqdm(valid_ds):
            input, target = input.to(device), target.to(device)
            output = model(input)
            val_loss = criterion(output, target)

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


def test_epoch_ensemble(
    model: nn.Module,
    valid_ds: DataLoader,
    criterion: nn.CrossEntropyLoss
):
    device = get_device()

    val_losses = []
    val_accuracies = []
    batch_sizes = []
    predictions = []
    targets = []

    with torch.no_grad():
        for input, target in tqdm(valid_ds):
            input, target = input.to(device), target.to(device)
            output = model(input)
            val_loss = criterion(output, target)

            # Calculate accuracy
            pred = torch.argmax(output, dim=-1)
            accurate_pred = (pred == target).type(torch.float32)
            val_accuracies.append(torch.mean(accurate_pred).item())

            val_losses.append(val_loss.item())
            # Store information regarding
            batch_sizes.append(len(input))

            predictions.append(pred)
            targets.append(target)

    val_accuracy = np.average(val_accuracies, weights=batch_sizes)
    val_loss = np.average(val_losses, weights=batch_sizes)
    predictions = torch.concatenate(predictions).cpu().numpy()
    targets = torch.concatenate(targets).cpu().numpy()
    return val_accuracy, val_loss, targets, predictions


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_path", required=True)
    return parser.parse_args()

########### 

def test_final_classification(config_path: str, data_path: str):
    config = None
    with open(config_path) as f:
        config_str = f.read()
        config = json.loads(config_str)

    criterion = nn.CrossEntropyLoss()

    checkpoint = config["checkpoint_folder"]
    os.makedirs(checkpoint, exist_ok=True)

    with open(os.path.join(checkpoint, "config.json"), "w") as f:
        f.write(json.dumps(config))

    batch_size = int(config["batch_size"]) if "batch_size" in config else 256

    _, _, cinic_test = get_cinic_ensemble(data_path, batch_size=batch_size)

    classes = cinic_test.dataset.classes
    classes = sorted(classes, key=lambda x: cinic_test.dataset.class_to_idx[x])

    print("Classes:")
    print(classes)

    # Calculate test metrics and confusion matrix
    test_accuracy, test_targets, test_predictions = test_epoch_ensemble_final_classification(cinic_test, criterion)
    with open(os.path.join(checkpoint, "test_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}")

    print("Test targets")
    print(test_targets)
    print("Test pred")
    print(test_predictions)

    # Save Scikit-learn report
    report = classification_report(test_targets, test_predictions, target_names=classes)
    with open(os.path.join(checkpoint, "classification_report.txt"), "w") as f:
        f.write(report)

    print(report)

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

   
def final_classification_batch(inputs: torch.Tensor,
                               binary_model: torch.nn.Module,
                               animals_model: torch.nn.Module,
                               vehicles_model: torch.nn.Module,
                               device: torch.device) -> torch.Tensor:
    # Run the binary classifier.
    binary_output = binary_model(inputs.to(device))
    binary_pred = torch.argmax(binary_output, dim=1)  # shape: [B]

    # Prepare a container for final predictions.
    final_pred = torch.empty_like(binary_pred)

    # Identify indices for each branch.
    animal_indices = (binary_pred == 0).nonzero(as_tuple=True)[0]
    vehicle_indices = (binary_pred == 1).nonzero(as_tuple=True)[0]

    # Process animal samples.
    if animal_indices.numel() > 0:
        animal_inputs = inputs[animal_indices].to(device)
        animal_output = animals_model(animal_inputs)
        animal_pred = torch.argmax(animal_output, dim=1)
        # Remap animal predictions: 0->2, 1->3, 2->4, 3->5, 4->6, 5->7.
        animal_mapping_tensor = torch.tensor([2, 3, 4, 5, 6, 7], device=device)
        animal_pred_orig = animal_mapping_tensor[animal_pred]
        final_pred[animal_indices] = animal_pred_orig

    # Process vehicle samples.
    if vehicle_indices.numel() > 0:
        vehicle_inputs = inputs[vehicle_indices].to(device)
        vehicle_output = vehicles_model(vehicle_inputs)
        vehicle_pred = torch.argmax(vehicle_output, dim=1)
        # Remap vehicle predictions: 0->0, 1->1, 2->8, 3->9.
        vehicle_mapping_tensor = torch.tensor([0, 1, 8, 9], device=device)
        vehicle_pred_orig = vehicle_mapping_tensor[vehicle_pred]
        final_pred[vehicle_indices] = vehicle_pred_orig
    return final_pred

def test_epoch_ensemble_final_classification(valid_ds, criterion):
    device = get_device()  # Assumes get_device() is defined elsewhere

    # Define checkpoint paths (adjust these paths as needed)
    binary_ckpt_path = "checkpoints/binary_classifier/state.pth"
    animals_ckpt_path = "checkpoints/animals_classifier/state.pth"
    vehicles_ckpt_path = "checkpoints/vehicles_classifier/state.pth"

    # Define model parameters for each branch.
    binary_model_params = {"num_classes": 2}
    animals_model_params = {"num_classes": 6}   
    vehicles_model_params = {"num_classes": 4}   

    # Import the model constructor from your models_ensemble module.
    from models_ensemble import EfficientNet_Transfer_Learning as model_constructor

    def load_model(ckpt_path, model_params):
        model = model_constructor(**model_params).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model

    # Load the three ensemble models.
    binary_model = load_model(binary_ckpt_path, binary_model_params)
    animals_model = load_model(animals_ckpt_path, animals_model_params)
    vehicles_model = load_model(vehicles_ckpt_path, vehicles_model_params)

    total_correct = 0
    total_samples = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, target in tqdm(valid_ds, desc="Evaluating ensemble"):
            inputs, target = inputs.to(device), target.to(device)
            batch_size = inputs.size(0)

            # Get final predictions for the batch using ensemble logic.
            final_pred = final_classification_batch(inputs, binary_model, animals_model, vehicles_model, device)

            total_correct += (final_pred == target).sum().item()
            total_samples += batch_size

            predictions.append(final_pred)
            targets.append(target)

    overall_accuracy = total_correct / total_samples
    predictions = torch.cat(predictions).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    return overall_accuracy, targets, predictions

if __name__ == "__main__":
    args = get_arguments()
    train_ensemble_classifier(args.config, args.data_path)