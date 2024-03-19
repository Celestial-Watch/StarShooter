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


def train_one_epoch(
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
        loss = criterion(preds, labels.float())
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


def report_performance(
    logger: tensorboard.writer.SummaryWriter,
    training_loss: float,
    val_loss: float,
    val_accuracy: float,
    epoch: int,
) -> None:
    # Printing
    print("LOSS train {} valid {}".format(training_loss, val_loss))
    print("VALIDATION-ACCURACY: ", val_accuracy * 100, "%")

    # Loggingn
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


def train(
    model: nn.Module,
    training_loader: torch.utils.data.DataLoader,
    val_dataset: torch.utils.data.TensorDataset,
    criterion: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    experiment_name: str,
):
    # Create tensorboard logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "{}_{}".format(experiment_name, timestamp)
    logger = tensorboard.writer.SummaryWriter("logs/" + run_name)

    model_folder = "model/{}".format(run_name)
    os.makedirs(model_folder)

    # Needed to compute validation accuracy
    val_images, val_labels = val_dataset[:]
    best_val_loss = torch.inf
    best_model = model
    for epoch in range(num_epochs):
        print("EPOCH {}:".format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        training_loss = train_one_epoch(
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


def train_basic(
    model: nn.Module,
    training_loader: torch.utils.data.DataLoader,
    val_dataset: torch.utils.data.TensorDataset,
    criterion: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    experiment_name: str,
):
    # Create tensorboard logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "{}_{}".format(experiment_name, timestamp)
    logger = tensorboard.writer.SummaryWriter("logs/" + run_name)

    model_folder = "model/{}".format(run_name)
    os.makedirs(model_folder)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Needed to compute validation accuracy
    val_images, val_labels = val_dataset[:]
    best_val_loss = torch.inf
    best_model = model
    for epoch in range(num_epochs):
        print("EPOCH {}:".format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        running_loss = 0.0
        for i, data in enumerate(training_loader):
            inputs, labels = data
            optimizer.zero_grad()  # reset
            preds = model(inputs)

            # Compute the loss and its gradients
            loss = criterion(preds, labels.float())
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            scheduler.step()

            # Gather data and report
            # print(loss.item())

            running_loss += loss.item()
            if i % 50 == 49:
                last_loss = running_loss / 50  # loss per batch

                # Log data
                print("  batch {} loss: {}".format(i + 1, last_loss))
                batch_number = epoch * len(training_loader) + i + 1
                logger.add_scalar("Loss/train", last_loss, batch_number)

                running_loss = 0.0

        val_loss, val_accuracy = get_validation_performance(
            model, val_images, val_labels, criterion
        )

        report_performance(logger, running_loss, val_loss, val_accuracy, epoch)

        # Track best performance, and save the model's state
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            model_path = model_folder + "/model_{}_{}".format(timestamp, epoch)
            torch.save(best_model.state_dict(), model_path)

    return best_model


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
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
    epochs = 10
    batch_size = 4
    experiment1 = "smol_image"
    experiment2 = f"big_image_{crop_size}_x_{crop_size}"

    # Load data
    small_image_set, big_image_set = get_datasets(crop_size)

    # train_loader1, val_loader1 = get_loaders(small_image_set, batch_size=batch_size)
    train_loader2, val_loader2 = get_loaders(big_image_set, batch_size=batch_size)

    # plot the first image in the trainloader with matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision

    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(train_loader2)
    images, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))

    print(f"Training on {len(train_loader2) * batch_size} samples.")
    # model1 = train(
    #     model1,
    #     train_loader1,
    #     val_loader1,
    #     loss,
    #     optimizer1,
    #     epochs,
    #     experiment1,
    # )
    model2 = train_basic(
        model2,
        train_loader2,
        val_loader2,
        loss,
        optimizer2,
        epochs,
        experiment2,
    )
