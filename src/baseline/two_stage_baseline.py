from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from multilayer_perceptron import MLP


class TwoStage(nn.Module):

    def __init__(self):
        super(TwoStage, self).__init__()
        self.stage1 = resnet18(weights = ResNet18_Weights.DEFAULT)
        self.stage2 = MLP()

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return x
