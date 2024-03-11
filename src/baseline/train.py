import copy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from datetime import datetime
from datetime import datetime
import os
import sys
from two_stage_baseline import TwoStage
from channel_model import ChannelResNet
from torch.utils import tensorboard
from typing import Tuple
from utils import get_dataframe, get_dataset, get_loaders, get_dataset_stage1



def train(model: nn.Module,
    training_loader: torch.utils.data.DataLoader,
    val_dataset: torch.utils.data.TensorDataset,
    criterion: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    experiment_name: str,
    two_stage_training: bool = False,
    ):

    # Create tensorboard logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "{}_{}".format(experiment_name, timestamp)
    logger = tensorboard.writer.SummaryWriter("logs/" + run_name)

    model_folder = "model/{}".format(run_name)
    os.makedirs(model_folder)

    # Needed to compute validation accuracy
    val_images, val_labels = val_dataset[:]

    best_val_loss = 1_000_000.0
    best_model = model
    for epoch in range(num_epochs):
        print("EPOCH {}:".format(epoch + 1))

        model.train()
        if two_stage_training:
            model.stage1.eval()

        # Make sure gradient tracking is on, and do a pass over the data
        training_loss = train_epoch(
            model, epoch, logger, criterion, training_loader, optimizer
        )

        val_loss, val_accuracy = get_validation_performance(
            model, val_images, val_labels, criterion
        )

        report_performance(logger, training_loss, val_loss, val_accuracy, epoch)

                # Track best performance, and save the model's state
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            model_path = model_folder + "/model_{}_{}".format(timestamp, epoch)
            torch.save(best_model.state_dict(), model_path)

    return best_model

def train_epoch(
    model: nn.Module,
    epoch_index: int,
    logger: tensorboard.writer.SummaryWriter,
    criterion: nn.modules.loss._Loss,
    training_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    window_size: int = 50,
):
    running_loss = 0.0
    last_loss = 0.0

    # Make sure at least one window is used. Used for inter-epoch reporting
    window_size = min(window_size, len(training_loader))
    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()  # reset
        preds = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(preds, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % window_size == window_size - 1:
            last_loss = running_loss / window_size  # loss per batch

                    # Log data
            print("  batch {} loss: {}".format(i + 1, last_loss))
            batch_number = epoch_index * len(training_loader) + i + 1
            logger.add_scalar("Loss/train", last_loss, batch_number)

            running_loss = 0.0

    return last_loss




# more code theivery, will change when other branch is pushed to master
# can get this moved to a utils file
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
        _, predicted_labels = torch.max(val_preds, 1)
        val_accuracy = (predicted_labels == val_labels).float().mean().item()
        val_loss = criterion(val_preds, val_labels).item()

    model.train()
    return val_loss, val_accuracy

# Robin's logging code
def report_performance(
    logger: tensorboard.writer.SummaryWriter,
    training_loss: float,
    val_loss: float,
    val_accuracy: float,
    epoch: int,
) -> None:
    # Printing
    print("LOSS train {} valid {}".format(training_loss, val_loss))
    print("ACCURACY: ", val_accuracy * 100, "%")

    # Logging
    logger.add_scalars(
        "Training vs. Validation Loss",
        {"Training": training_loss, "Validation": val_loss},
        epoch + 1,
    )
    logger.add_scalars(
        "Validation Accuracy",
        {"Accuracy": val_accuracy},
        epoch + 1,
    )
    logger.flush()


if __name__ == '__main__':

    input_dim = (30,30)

    # Load data
    path_to_data = os.path.abspath("./../processing/data/alistair/30x30_images/") + "/"
    path_to_csv = os.path.abspath("./../processing/data/alistair") + "/"
    dataset_stage1 = get_dataset_stage1(path_to_csv + "annotations.csv", path_to_data)
    train_loader_stage1, val_loader_stage1 = get_loaders(dataset_stage1)

    loss1 = nn.CrossEntropyLoss()


    stage1 = ChannelResNet(no_classes=8)
    optimiser = optim.Adam(stage1.parameters())
    stage1 = train(stage1, train_loader_stage1, val_loader_stage1, loss1, optimiser, 100, "stage1")

    movers_agg = get_dataframe(path_to_data + "csv/")

    dataset_stage2 = get_dataset(movers_agg, path_to_data, image_shape=input_dim)
    train_loader, val_loader = get_loaders(dataset_stage2)

    loss2 = nn.BCELoss()

    model = TwoStage(stage1)
    optimiser2 = optim.Adam(model.parameters())
    model = train(model, train_loader, val_loader, loss2, optimiser2, 100, "stage1", True)
