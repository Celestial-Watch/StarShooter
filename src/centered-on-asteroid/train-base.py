import model_def
import torch
import torch.nn as nn
import os
from typing import Tuple
from datetime import datetime
from torch.utils import tensorboard
import copy
import sys
from utils import get_stacked_dataset, get_loaders, get_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baseline.utils import get_dataframe


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

    # Start counts to determine precision, recall, and F1 score
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        val_preds = model(val_images)
        val_accuracy = ((val_preds > 0.5) == val_labels).float().mean().item()
        val_loss = criterion(val_preds, val_labels.float()).item()

        # determine true/false positives/negatives
        true_pos += ((val_preds > 0.5) & (val_labels == 1)).sum().item()
        false_pos += ((val_preds > 0.5) & (val_labels == 0)).sum().item()
        true_neg += ((val_preds <= 0.5) & (val_labels == 0)).sum().item()
        false_neg += ((val_preds <= 0.5) & (val_labels == 1)).sum().item()

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = 2 * (precision * recall) / (precision + recall)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 Score: ", f1)

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
    print("ACCURACY: ", val_accuracy * 100, "%")

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

    best_val_loss = 1_000_000.0
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


if __name__ == "__main__":
    # Model parameters
    image_shape = (30, 30)
    images_per_sequence = 4
    feature_vector_size = 10

    model = model_def.CFN(4, 10, (30, 30))

    # Training parameters
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 10
    batch_size = 4
    expiremt_name = "cfn-base"

    # Load data
    path_to_data = os.path.abspath("./../processing/data/alistair/30x30_images") + "/"
    path_to_csv = os.path.abspath("./../processing/data") + "/"
    movers_agg = get_dataframe(path_to_csv)
    data_set, _ = get_dataset(movers_agg, path_to_data)
    train_loader, val_loader = get_loaders(data_set, batch_size=batch_size)

    print(f"Training on {len(train_loader) * batch_size} samples.")
    model = train(
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        epochs,
        expiremt_name,
    )
