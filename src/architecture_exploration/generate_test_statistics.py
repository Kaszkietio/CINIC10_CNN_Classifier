import os
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

sys.path.append("../src")

from data_prep import get_cinic
from utils import get_device, set_seed
from custom_models import ResNet_32x32, AlexNet_32x32


ARCHITECTURES = {
    "resnet": ResNet_32x32,
    "alexnet": AlexNet_32x32,
}


def generate_test_statistics(ds_test, model_path, architecture, seed):

    print("Generating statistic for", model_path)
    set_seed(seed)
    model: nn.Module = ARCHITECTURES[architecture]().cuda()
    state_dict = torch.load(os.path.join(model_path, "state.pth"), map_location=get_device())
    model.load_state_dict(state_dict["model_state"])

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for input, target in tqdm(ds_test):
            input, target = input.cuda(), target

            outputs = model(input)
            _, predictions = torch.max(outputs, 1)

            y_true.extend(target.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)


    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    with open(model_path + "test_statistics.csv", "w") as f:
        f.write("accuracy,precision,recall,f1_score\n")
        f.write(f"{accuracy},{precision},{recall},{f1}\n")
    print(f"Test statistics for {model_path}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("============== Done ===============")


if __name__ == "__main__":
    _, _, ds_test = get_cinic("../data")
    to_process = [
        ('ResNet_sgd_mom_0_9', "resnet", 1),
        ('ResNet_sgd_mom_0_99', "resnet", 1),
        ('ResNet_wdec_0', "resnet", 1),
        ('ResNet_wdec_0_01', "resnet", 1),
        ('ResNet_wdec_0_1', "resnet", 1),
        ('ResNet_wdec_0_4', "resnet", 1)
    ]
    for model_path, architecture, seed in to_process:
        generate_test_statistics(ds_test, f"../checkpoints/{model_path}/", architecture, seed)
