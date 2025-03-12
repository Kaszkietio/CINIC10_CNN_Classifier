import torch
from torch import nn
import torch.nn.functional as F

class AlexNet_32x32(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()
        # Convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.conv3 = nn.Conv2d(192, 256, kernel_size=3, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.relu = nn.ReLU(inplace=True)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classification
        self.dropout1 = nn.Dropout(dropout)
        self.lin1 = nn.Linear(256 * 4 * 4, 4096)

        self.dropout2 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(4096, 4096)

        self.dropout3 = nn.Dropout(dropout)
        self.lin3 = nn.Linear(4096, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool1(self.relu(self.conv1(x)))
        x = self.maxpool2(self.relu(self.conv2(x)))
        x = self.maxpool3(self.relu(self.conv3(x)))
        x = self.maxpool4(self.relu(self.conv4(x)))

        x = self.global_avg_pool(x)

        x = torch.flatten(x, 1)

        x = self.lin1(self.dropout1(x))
        x = self.lin2(self.dropout2(x))
        x = self.lin3(self.dropout3(x))

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsampler = None):
        super().__init__()
        self.downsampler = downsampler

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x
        if self.downsampler is not None:
            identity = self.downsampler(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet_32x32(nn.Module):
    def __init__(self, num_classes: int = 10, block: nn.Module = ResidualBlock):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        self.bnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.res_layer1 = self._residual_layer(block, 64, 128, 2)

        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.res_layer2 = self._residual_layer(block, 128, 256, 2)

        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.res_layer3 = self._residual_layer(block, 256, 512, 2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.lin = nn.Linear(512, num_classes)

    def _residual_layer(self, block, in_channels, out_channels, blocks_num, stride=1):
        """Creates a residual layer consisting out of residual blocks"""
        downsampler = None
        if stride != 1 or in_channels != out_channels:
            downsampler = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsampler))
        for _ in range(blocks_num - 1):
            layers.append(block(out_channels, out_channels, stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bnorm(self.conv(x)))

        x = self.res_layer1(self.max_pool1(x))
        x = self.res_layer2(self.max_pool2(x))
        x = self.res_layer3(self.max_pool3(x))

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.lin(x)
        return x