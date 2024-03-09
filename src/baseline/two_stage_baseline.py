import torch.nn as nn
import torch
from torchvision.models import resnet18, ResNet18_Weights
from typing import Tuple


class TwoStage(nn.Module):

    def __init__(self,
        images_per_sequence: int,
        no_classes: int,
        image_shape: Tuple[int, int],
        stage_1: nn.Module
        ):

        super(TwoStage, self).__init__()
        self.stage1 = stage_1
        self.stage1.eval()
        self.stage2 = MLP()

    def forward(self, images) -> torch.Tensor:

        output_classes = self.stage1(images)
        x = self.stage2(output_classes)

        return x
    

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class Stage1(nn.Module):
    def __init__(self,
        images_per_sequence: int,
        no_classes: int,
        image_shape: Tuple[int, int],
        ):
        super(Stage1, self).__init__()
        self.image_shape = image_shape
        self.images_per_sequence = images_per_sequence
        self.no_classes = no_classes
        self.resnet = ResNetClassifer(no_classes)

    def forward(self, x) -> torch.Tensor:
        images = torch.split(x, self.image_shape[0], dim=2)
        classes = [self.resnet(image) for image in images]
        x = torch.cat(classes, dim=1)
        return x


class ResNetClassifer(nn.Module):
    def __init__(self, no_classes: int):
        super(ResNetClassifer, self).__init__()
        self.resnet = resnet18(weights = ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, no_classes)

    def forward(self, x) -> torch.Tensor:
        return self.resnet(x)
