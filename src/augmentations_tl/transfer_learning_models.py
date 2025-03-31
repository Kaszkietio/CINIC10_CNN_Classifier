import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNet_Transfer_Learning_v1(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier[1] = nn.Linear(in_features = self.model.classifier[1].in_features, out_features = num_classes)

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        print(f"Model structure: {self.model}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class EfficientNet_Transfer_Learning_v2(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        for name, param in self.model.named_parameters():
            if "features.7" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.model.classifier[1] = nn.Linear(in_features = self.model.classifier[1].in_features, out_features = num_classes)

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        print(f"Model structure: {self.model}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

class EfficientNet_Transfer_Learning_v3(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        for name, param in self.model.named_parameters():
            if "features.6" in name or "features.7" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.model.classifier[1] = nn.Linear(in_features = self.model.classifier[1].in_features, out_features = num_classes)

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        print(f"Model structure: {self.model}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
