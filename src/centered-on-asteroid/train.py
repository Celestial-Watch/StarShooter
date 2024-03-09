import model_def
import torch
import torch.nn as nn
import os
from typing import Tuple
from datetime import datetime
from torch.utils import tensorboard
import copy
import os
from utils import (
    get_dataframe,
    get_dataset,
    get_loaders,
    get_position_tensor,
    get_engineered_features,
    CustomDataset,
)
from argparse import ArgumentParser


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


def get_experiment_args():
    parser = ArgumentParser()
    parser.add_argument("--image_shape", type=int, nargs=2, default=(30, 30))
    parser.add_argument("--images_per_sequence", type=int, default=4)
    parser.add_argument("--feature_vector_size", type=int, default=10)
    parser.add_argument("--loss", type=str, default="BCELoss")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--experiment_name", type=str, default="end-to-end")
    parser.add_argument("--metadata", type=str, default="None")
    return parser.parse_args()


if __name__ == "__main__":
    # Loss lookup table
    classification_loss_functions = {
        "CrossEntropyLoss": nn.CrossEntropyLoss(),
        "NLLLoss": nn.NLLLoss(),
        "BCELoss": nn.BCELoss(),
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
        "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss(),
        "MultiMarginLoss": nn.MultiMarginLoss(),
        "SoftMarginLoss": nn.SoftMarginLoss(),
    }
    args = get_experiment_args()
    # Model parameters
    image_shape = args.image_shape
    images_per_sequence = args.images_per_sequence
    feature_vector_size = args.feature_vector_size

    # Load data
    # Ignore weird paths for now
    path_to_data = os.path.abspath("./../../scripts/data")
    real_movers_file = f"{path_to_data}/csv/csv/movers_cond_12_image_meta_data.csv"
    bogus_movers_file = f"{path_to_data}/csv/csv/movers_cond_2_image_meta_data.csv"
    images_folder = f"{path_to_data}/images/centered_on_asteroid/"
    movers_agg = get_dataframe(real_movers_file, bogus_movers_file)
    data_set, mover_ids = get_dataset(movers_agg, images_folder)

    # Metadata
    metadata = args.metadata
    if metadata != "None":
        # Get engineered features -- TODO: Add other metadata options
        engineered_features = "positions"
        movers_agg = movers_agg.filter(
            lambda x: any(x["mover_id"].isin(mover_ids))
        ).groupby("mover_id")
        metadata = get_position_tensor(movers_agg)
        extra_features = get_engineered_features(metadata, engineered_features)

        data_set = CustomDataset(
            data_set.tensors[0], extra_features, data_set.tensors[1]
        )

        metadata_size = len(extra_features[0])
        model = model_def.MCFN(
            images_per_sequence, feature_vector_size, image_shape, metadata_size
        )
    else:
        model = model_def.CFN(images_per_sequence, feature_vector_size, image_shape)

    # Training parameters
    loss = classification_loss_functions[args.loss]
    optimizer = torch.optim.Adam(model.parameters())
    epochs = args.epochs
    batch_size = args.batch_size
    experiment_name = args.experiment_name

    train_loader, val_loader = get_loaders(data_set, batch_size=batch_size)

    print(f"Val loader: {val_loader}")

    print(f"Training on {len(train_loader)} samples.")
    model = train(
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        epochs,
        experiment_name,
    )
