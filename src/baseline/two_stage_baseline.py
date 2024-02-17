import torch.nn as nn
import torch
from multilayer_perceptron import MLP
from resnet import Stage1
from torchvision.models import resnet18, ResNet18_Weights


class TwoStage(nn.Module):

    def __init__(self):

        super(TwoStage, self).__init__()
        # self.stage1 = Stage1(weights = Stage1.Weights)
        self.stage1 = resnet18(weights = ResNet18_Weights.DEFAULT)
        self.stage1.fc = nn.Linear(self.stage1.fc.in_features, 8)
        self.stage2 = MLP()

    def forward(self, images) -> torch.Tensor:

        output = []
        for image in images:
            output = output + self.stage1(image)
        x = self.stage2(output)

        return x
