import channel_model
import torch
import torch.nn as nn
import os
import sys
from typing import Tuple
from datetime import datetime
from torch.utils import tensorboard
import copy
from centre_images import get_loaders, get_datasets

sys.path.insert(1, os.path.join(os.path.dirname(__file__), "../centered-on-asteroid"))
from train import train

if __name__ == "__main__":
    # Model parameters
    image_shape1 = 30
    image_shape2 = 100
    crop_size = image_shape2
    images_per_sequence = 4

    model1 = channel_model.ChannelResNet(image_shape1)
    model2 = channel_model.ChannelResNet(image_shape2)

    # Training parameters
    loss = torch.nn.BCELoss()
    optimizer1 = torch.optim.Adam(model1.parameters())
    optimizer2 = torch.optim.Adam(model2.parameters())
    epochs = 10
    batch_size = 4
    experiment1 = "smol_image"
    experiment2 = "big_image"

    # Load data
    small_image_set, big_image_set = get_datasets(crop_size)

    train_loader1, val_loader1 = get_loaders(small_image_set, batch_size=batch_size)
    train_loader2, val_loader2 = get_loaders(big_image_set, batch_size=batch_size)

    print(f"Training on {len(train_loader1) * batch_size} samples.")
    model1 = train(
        model1,
        train_loader1,
        val_loader1,
        loss,
        optimizer1,
        epochs,
        experiment1,
    )
    model2 = train(
        model2,
        train_loader2,
        val_loader2,
        loss,
        optimizer2,
        epochs,
        experiment2,
    )
