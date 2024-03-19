import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelResNet(nn.Module):
    def __init__(
        self,
        image_size,
    ):
        super(ChannelResNet, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.PReLU()

        self.residual_block1 = self._make_residual_block(32, 32, 2)

        self.fc = nn.Linear(32 * image_size * image_size, 1)

    def _make_residual_block(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.residual_block1(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out
