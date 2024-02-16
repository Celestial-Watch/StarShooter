import model_def
import torch
import torch.nn as nn
import pandas as pd
import torchvision
import os
from PIL import Image
from typing import List, Tuple
from datetime import datetime
from torch.utils import tensorboard
import copy
import os
import pandas.api.typing as pd_typing


def get_dataframe(path_to_csvs: str) -> pd.DataFrame:
    """
    Returns a DataFrame object grouped by the mover_id with columns file_name and label
    """
    # Read csv
    real_movers = pd.read_csv(path_to_csvs + "movers_images_lookup.csv")
    bogus_movers = pd.read_csv(path_to_csvs + "rejected_movers_images_lookup.csv")

    # Add labels
    real_movers["label"] = 1
    bogus_movers["label"] = 0

    # Group by mover
    movers = pd.concat([real_movers, bogus_movers])
    movers_agg = movers.groupby("mover_id")
    return movers_agg


def get_dataset(
    movers_agg: pd_typing.DataFrameGroupBy,
    path_to_images: str,
    image_shape: tuple = (30, 30),
) -> torch.utils.data.TensorDataset:
    # Generate input, output pairs
    x_tensors = []
    y_hat_tensors = []
    for mover_id, group_data in movers_agg:
        image_tensors = []
        # Ignore sequences that aren't 4 images long
        if len(group_data) != 4:
            print(f"Skipping {mover_id} sequence with length: {len(group_data)}")
            continue

        for _, row in group_data.iterrows():
            image_path = path_to_images + row["file_name"]
            try:
                # Read image as PIL Image and convert to grayscale
                image = Image.open(image_path).convert("L")
            except FileNotFoundError:
                print(f"Image of {mover_id} not found: {image_path}")
                break

            # Convert PIL Image to torch.Tensor
            transform = torchvision.transforms.ToTensor()
            image_tensor = transform(image)

            # Reshape image tensor to match the expected input shape
            image_tensor = image_tensor.view(1, 1, *image_shape)
            image_tensors.append(image_tensor)
        else:
            # Loop finished without break
            # Concatenate over width dimension -> (1, 1, 120, 30)
            x_tensor = torch.cat(image_tensors, dim=2)
            x_tensors.append(x_tensor)
            y_hat_tensors.append(torch.tensor([[group_data["label"].iloc[0]]]))

    x = torch.concat(x_tensors)
    y_hat = torch.concat(y_hat_tensors)
    data_set = torch.utils.data.TensorDataset(x, y_hat)
    return data_set


def get_loaders(
    data_set: torch.utils.data.TensorDataset,
    split: List[float] = [0.7, 0.3],
    batch_size: int = 4,
):
    train_data_set, val_data_set = torch.utils.data.random_split(data_set, split)
    train_loader = torch.utils.data.DataLoader(
        train_data_set, batch_size=batch_size, shuffle=True
    )
    return train_loader, val_data_set


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

    model = model_def.MNN(images_per_sequence, feature_vector_size, image_shape)

    # Training parameters
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 100
    batch_size = 4
    expiremt_name = "end-to-end"

    # Load data
    path_to_data = os.path.abspath("./../../data/") + "/"
    movers_agg = get_dataframe(path_to_data + "csv/")
    data_set = get_dataset(movers_agg, path_to_data + "30x30_images/")
    train_loader, val_loader = get_loaders(data_set, batch_size=batch_size)

    model = train(
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        epochs,
        expiremt_name,
    )
