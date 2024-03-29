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
from train import *

if __name__ == "__main__":
    # Fixed hyper parameters
    loss = torch.nn.BCELoss()
    epochs = 10
    batch_size = 4
    images_per_sequence = 4

    # Hyper parameters to search
    num_conv_blocks = 2
    kernel_size = 5
    num_filters = 32
    feature_vector_size = 10
    learning_rate = 0.0005
    mlp_hidden_layers = 2
    mlp_hidden_units = 64

    # Define the grid search
    grid_search = {
        "num_conv_blocks": num_conv_blocks,
        "kernel_size": kernel_size,
        "num_filters": num_filters,
        "feature_vector_size": feature_vector_size,
        "learning_rate": learning_rate,
        "mlp_hidden_layers": mlp_hidden_layers,
        "mlp_hidden_units": mlp_hidden_units,
    }

    # Load data
    real_movers_file = "../../data/csv/movers_image_metadata.csv"
    bogus_movers_file = "../../data/csv/rejected_movers_image_metadata.csv"
    images_folder = "../../data/centered_on_asteroids/"
    movers_agg = get_dataframe(real_movers_file, bogus_movers_file)
    raw_data_set, mover_ids = get_dataset(movers_agg, images_folder)

    # list_engineer_features = ["max_grad_diff", "max_ang_diff","max_movement_vector_distance",  "max_movement_vector_distance_normalised", "gradients",  "angles", "movement_vectors", "positions"]
    list_engineer_features = ["no_metadata"]
    for feature in list_engineer_features:

        # Get engineered features
        movers_agg = movers_agg.filter(
            lambda x: any(x["mover_id"].isin(mover_ids))
        ).groupby("mover_id")
        metadata = get_position_tensor(movers_agg)
        extra_features = get_engineered_features(metadata, feature)

        data_set = CustomDataset(
            raw_data_set.tensors[0], extra_features, raw_data_set.tensors[1]
        )

        # Define the model
        image_shape = (30, 30)
        metadata_size = len(extra_features[0])

        # Loop through the hyper parameters
        # for num_conv_blocks in grid_search["num_conv_blocks"]:
        #     for kernel_size in grid_search["kernel_size"]:
        #         for num_filters in grid_search["num_filters"]:
        #             for feature_vector_size in grid_search["feature_vector_size"]:
        #                 for mlp_hidden_layers in grid_search["mlp_hidden_layers"]:
        #                     for mlp_hidden_units in grid_search["mlp_hidden_units"]:
        #                         for learning_rate in grid_search["learning_rate"]:

        # filters list
        filters = [num_filters] * num_conv_blocks

        model = model_def.DynamicCFN(
            image_shape=image_shape,
            num_conv_blocks=num_conv_blocks,
            conv_filters_list=filters,
            conv_kernel_size=kernel_size,
            feature_vector_output_size=feature_vector_size,
            images_per_sequence=images_per_sequence,
            metadata_size=metadata_size,
            hidden_mlp_layers=mlp_hidden_layers,
            hidden_mlp_size=mlp_hidden_units,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Define the data loaders
        train_loader, val_data_set = get_loaders(data_set, batch_size=batch_size)

        # Experiment Name
        experiment_name = f"{feature}_best_model"

        # Print training config
        print(f"Training model with parameters: {experiment_name}")
        print(f"Number of Conv Blocks: {num_conv_blocks}")
        print(f"Kernel Size: {kernel_size}")
        print(f"Number of Filters: {num_filters}")
        print(f"Feature Vector Size: {feature_vector_size}")
        print(f"Learning Rate: {learning_rate}")
        print(f"MLP Hidden Layers: {mlp_hidden_layers}")
        print(f"MLP Hidden Units: {mlp_hidden_units}")
        # print(f"Model: {model}")

        # Train the model
        train(
            model,
            train_loader,
            val_data_set,
            loss,
            optimizer,
            epochs,
            experiment_name,
        )
