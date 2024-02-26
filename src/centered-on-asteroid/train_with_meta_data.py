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

    model = model_def.MCFN(images_per_sequence, feature_vector_size, image_shape, 1)

    # Training parameters
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 10
    batch_size = 4
    expiremt_name = "end-to-end"

    # Load data
    path_to_data = os.path.abspath("./../../data/")
    real_movers_file = f"{path_to_data}/csv/movers_cond_12_image_meta_data.csv"
    bogus_movers_file = f"{path_to_data}/csv/movers_cond_2_image_meta_data.csv"
    images_folder = f"{path_to_data}/images/centered_on_asteroid/"
    movers_agg = get_dataframe(real_movers_file, bogus_movers_file)
    data_set, _ = get_dataset(movers_agg, images_folder)
    metadata = get_position_tensor(movers_agg)
    # metadata = torch.fill(metadata, 0)
    extra_features = get_engineered_features(metadata)

    # plt.scatter(
    #     extra_features[:], data_set.tensors[1].numpy(), c=data_set.tensors[1].numpy()
    # )
    # plt.xlabel("Feature 1")
    # plt.ylabel("Feature 2")
    # plt.title("Extra Features Colored by Label")
    # plt.colorbar()
    # plt.show()

    data_set = CustomDataset(data_set.tensors[0], extra_features, data_set.tensors[1])

    train_loader, val_loader = get_loaders(data_set, batch_size=batch_size)

    print(f"Training on {len(train_loader)*batch_size} samples.")
    model = train(
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        epochs,
        expiremt_name,
    )
