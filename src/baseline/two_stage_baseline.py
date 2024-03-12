import torch.nn as nn
import torch
from torchvision.models import resnet18, ResNet18_Weights
from typing import Tuple


class TwoStage(nn.Module):

    def __init__(self,
        stage_1: nn.Module
        ):

        super(TwoStage, self).__init__()
        self.stage1 = stage_1
        self.stage1.eval()
        self.stage2 = MLP()

    def forward(self, images) -> torch.Tensor:

        output_classes = []
        for i in range(images.shape[0]):
            output_classes.append(self.stage1(images[i]))
        output_classes = torch.concat(output_classes)
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
