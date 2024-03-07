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
import seaborn as sns
import pandas as pd


def visualize_max_grad_diff_dist(
    metadata: torch.Tensor, data_set: torch.utils.data.Dataset
):
    extra_features = get_engineered_features(metadata, type="max_grad_diff")

    combined_data = pd.DataFrame(
        {
            "extra_features": np.squeeze(extra_features.numpy()),
            "labels": np.squeeze(data_set.tensors[1].numpy()),
        }
    )
    n_bins = 10
    bin_width = 3
    bins = [bin_width * i for i in range(n_bins)]
    combined_data.loc[
        combined_data["extra_features"] > (n_bins - 2) * bin_width, "extra_features"
    ] = (n_bins - 2) * bin_width + 1
    ax = sns.histplot(
        combined_data,
        x="extra_features",
        hue="labels",
        bins=bins,
        kde=True,
        stat="density",
        multiple="dodge",
        common_norm=False,
    )
    labels = [str(i) for i in bins]
    labels[-1] = str((n_bins - 2) * bin_width) + "+"
    ax.set_xticks(bins)
    ax.set_xticklabels(labels)
    ax.legend(["Real Mover", "Bogus Mover"])
    plt.title("Distribution of max grad diff")
    plt.show()


def visualize_max_angle_diff_dist(
    metadata: torch.Tensor, data_set: torch.utils.data.Dataset
):
    extra_features = get_engineered_features(metadata, type="max_ang_diff")

    combined_data = pd.DataFrame(
        {
            "extra_features": np.squeeze(extra_features.numpy()),
            "labels": np.squeeze(data_set.tensors[1].numpy()),
        }
    )
    n_bins = 18
    bin_width = 10
    bins = [bin_width * i for i in range(n_bins + 1)]
    ax = sns.histplot(
        combined_data,
        x="extra_features",
        hue="labels",
        bins=bins,
        kde=True,
        stat="density",
        multiple="dodge",
        common_norm=False,
    )
    labels = [str(i) for i in bins]
    ax.set_xticks(bins)
    ax.set_xticklabels(labels)
    ax.legend(["Real Mover", "Bogus Mover"])
    plt.title("Distribution of max angle diff")
    plt.show()


def visualize_max_movement_vector_distance_dist(
    metadata: torch.Tensor, data_set: torch.utils.data.Dataset
):
    extra_features = get_engineered_features(
        metadata, type="max_movement_vector_distance"
    )

    combined_data = pd.DataFrame(
        {
            "extra_features": np.squeeze(extra_features.numpy()),
            "labels": np.squeeze(data_set.tensors[1].numpy()),
        }
    )
    n_bins = 20
    bin_width = 0.0001
    bins = [bin_width * i for i in range(n_bins)]
    combined_data.loc[
        combined_data["extra_features"] > (n_bins - 2) * bin_width, "extra_features"
    ] = (n_bins - 2) * bin_width + 1
    print(combined_data)
    ax = sns.histplot(
        combined_data,
        x="extra_features",
        hue="labels",
        bins=bins,
        kde=True,
        stat="density",
        multiple="dodge",
        common_norm=False,
    )
    labels = [str(i) for i in bins]
    labels[-1] = str((n_bins - 2) * bin_width) + "+"
    ax.set_xticks(bins)
    ax.set_xticklabels(labels)
    ax.legend(["Real Mover", "Bogus Mover"])
    plt.title("Distribution of max movement vector distance")
    plt.show()


def visualize_velocity_vectors(metadata, data_set):
    extra_features = get_engineered_features(metadata, type="movement_vectors")

    combined_data = pd.DataFrame(
        {
            "dx": [],
            "dy": [],
            "labels": [],
        }
    )
    for i in range(0, len(extra_features)):
        for j in range(0, len(extra_features[i]), 2):
            combined_data.loc[len(combined_data)] = [
                extra_features[i][j].item(),
                extra_features[i][j + 1].item(),
                data_set.tensors[1][i][0].item(),
            ]
    sns.scatterplot(data=combined_data, x="dx", y="dy", hue="labels")
    plt.title("Velocity vectors")
    plt.show()


if __name__ == "__main__":
    # Model parameters
    image_shape = (30, 30)
    images_per_sequence = 4
    feature_vector_size = 10

    model = model_def.MCFN(images_per_sequence, feature_vector_size, image_shape, 1)

    # Load data
    path_to_data = os.path.abspath("./../../data/")
    real_movers_file = f"{path_to_data}/csv/movers_cond_12_image_meta_data.csv"
    bogus_movers_file = f"{path_to_data}/csv/movers_cond_2_image_meta_data.csv"
    images_folder = f"{path_to_data}/images/centered_on_asteroid/"
    movers_agg = get_dataframe(real_movers_file, bogus_movers_file)
    data_set, mover_ids = get_dataset(movers_agg, images_folder)

    # Get engineered features
    movers_agg_filtered = movers_agg.filter(
        lambda x: any(x["mover_id"].isin(mover_ids))
    ).groupby("mover_id")
    metadata = get_position_tensor(movers_agg_filtered)
    # metadata = torch.fill(metadata, 0)
    visualize_max_grad_diff_dist(metadata, data_set)
    visualize_velocity_vectors(metadata, data_set)
    visualize_max_angle_diff_dist(metadata, data_set)
    visualize_max_movement_vector_distance_dist(metadata, data_set)
