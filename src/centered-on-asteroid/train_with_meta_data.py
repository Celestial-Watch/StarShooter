import model_def
import torch
import os
from utils import (
    get_dataframe,
    get_dataset,
    get_loaders,
    get_position_tensor,
    get_engineered_features,
)
from train import train
import matplotlib.pyplot as plt
import numpy as np


# Define a custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, metadata, labels):
        assert len(images) == len(metadata) == len(labels)
        self.images = images
        self.metadata = metadata
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (self.images[idx], self.metadata[idx]), self.labels[idx]


if __name__ == "__main__":
    print("Training model with meta data.")
    # Model parameters
    image_shape = (30, 30)
    images_per_sequence = 4
    feature_vector_size = 10

    engineered_features = "positions"
    expirement_name = f"end-to-end-{engineered_features}"

    # Load data
    path_to_data = os.path.abspath("./../../data/")
    real_movers_file = f"{path_to_data}/csv/movers_cond_12_image_meta_data.csv"
    bogus_movers_file = f"{path_to_data}/csv/movers_cond_2_image_meta_data.csv"
    images_folder = f"{path_to_data}/images/centered_on_asteroid/"
    movers_agg = get_dataframe(real_movers_file, bogus_movers_file)
    data_set, mover_ids = get_dataset(movers_agg, images_folder)

    # Get engineered features
    movers_agg = movers_agg.filter(
        lambda x: any(x["mover_id"].isin(mover_ids))
    ).groupby("mover_id")
    metadata = get_position_tensor(movers_agg)
    extra_features = get_engineered_features(metadata, engineered_features)

    data_set = CustomDataset(data_set.tensors[0], extra_features, data_set.tensors[1])

    metadata_size = len(extra_features[0])
    model = model_def.MCFN(
        images_per_sequence, feature_vector_size, image_shape, metadata_size
    )

    print(f"Model: {model}")
    # Training parameters
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 10
    batch_size = 4

    train_loader, val_data_set = get_loaders(data_set, batch_size=batch_size)

    print(f"Training on {len(train_loader)*batch_size} samples.")
    print(f"Validating on {len(val_data_set)} samples.")
    model = train(
        model,
        train_loader,
        val_data_set,
        loss,
        optimizer,
        epochs,
        expirement_name,
    )