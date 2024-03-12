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


# this function is altered to allow the model to be trained with classes for validation
def get_validation_performance(
    model: nn.Module,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    criterion: nn.modules.loss._Loss,
) -> Tuple[float, float]:
    # Turn off dropout and batch normalization
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        val_preds = model(val_images)
        val_accuracy = ((val_preds > 0.5) == val_labels).float().mean().item()
        val_loss = criterion(val_preds, val_labels.float()).item()

    return val_loss, val_accuracy


if __name__ == "__main__":
    # Model parameters
    image_shape1 = 30
    image_shape2 = 100
    crop_size = image_shape2
    images_per_sequence = 4

    model1 = channel_model.ChannelResNet()
    model2 = channel_model.ChannelResNet()

    # Training parameters
    loss = torch.nn.BCELoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=5e-4)
    optimizer2 = torch.optim.Adam(model2.parameters())
    epochs = 100
    batch_size = 20
    experiment1 = "smol_image"
    experiment2 = "big_image"

    # Load data
    small_image_set = get_datasets(crop_size)
    # small_image_set, big_image_set = get_datasets(crop_size)

    train_loader1, val_loader1 = get_loaders(small_image_set, batch_size=batch_size)
    # train_loader2, val_loader2 = get_loaders(big_image_set, batch_size=batch_size)

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
    # model2 = train(
    #     model2,
    #     train_loader2,
    #     val_loader2,
    #     loss,
    #     optimizer2,
    #     epochs,
    #     experiment2,
    # )
