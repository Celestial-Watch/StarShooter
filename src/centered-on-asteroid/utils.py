import torch
import pandas as pd
import pandas.api.typing as pd_typing
from typing import List
from PIL import Image
import torchvision
from typing import Tuple
import math
import numpy as np

x_cord = "pos_RightAscension"
y_cord = "pos_Declination"


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


def get_position_tensor(
    movers_agg: pd_typing.DataFrameGroupBy,
) -> torch.Tensor:
    movers_positions = []
    for mover_id, group_data in movers_agg:
        mover_positions = []
        for _, row in group_data.iterrows():
            if math.isnan(row[x_cord]) or math.isnan(row[y_cord]):
                print(f"Missing position data for {mover_id}")
                break
            mover_positions.append(row[x_cord])
            mover_positions.append(row[y_cord])
        movers_positions.append(torch.Tensor(mover_positions))
    return torch.stack(movers_positions)


def get_engineered_features(
    movers_positions: torch.Tensor, type: str = "max_grad_diff"
) -> torch.Tensor:
    """

    Args:
        movers_positions (torch.Tensor): (x, y) position for the 4 images. Shape: (n, 8)
        type (str): The type of engineered features to return. Options: "max_grad_diff", "gradients", "movement_vectors"

    Returns: The engineered features for the given type (n, z), where z is the feature vector size
    """
    match type:
        case "no_metadata":
            return torch.full((len(movers_positions), 1), 0)
        case "max_grad_diff":
            get_max_grad_diffs = []
            for mover_positions in movers_positions:
                get_max_grad_diffs.append(get_max_grad_diff(mover_positions))
            return torch.stack(get_max_grad_diffs)
        case "max_ang_diff":
            max_ang_diffs = []
            for mover_positions in movers_positions:
                max_ang_diffs.append(get_max_ang_diff(mover_positions))
            return torch.stack(max_ang_diffs)
        case "max_movement_vector_distance":
            max_movement_vector_distances = []
            for mover_positions in movers_positions:
                max_movement_vector_distances.append(
                    get_max_movement_vector_distance(mover_positions, False)
                )
            return torch.stack(max_movement_vector_distances)
        case "max_movement_vector_distance_normalised":
            max_movement_vector_distances = []
            for mover_positions in movers_positions:
                max_movement_vector_distances.append(
                    get_max_movement_vector_distance(mover_positions, True)
                )
            return torch.stack(max_movement_vector_distances)
        case "gradients":
            gradients = []
            for mover_positions in movers_positions:
                gradients.append(get_gradients(mover_positions))
            return torch.stack(gradients)
        case "angles":
            angles = []
            for mover_positions in movers_positions:
                angles.append(get_angles(mover_positions))
            return torch.stack(angles)
        case "movement_vectors":
            movement_vectors = []
            for mover_positions in movers_positions:
                movement_vectors.append(
                    torch.flatten(get_movement_vectors(mover_positions))
                )

            return torch.stack(movement_vectors)
        case "positions":
            return movers_positions
        case _:
            raise ValueError(f"Invalid type: {type}")


def get_max_movement_vector_distance(
    positions: torch.Tensor, normalise: bool
) -> torch.Tensor:
    """
    Returns the maximum distance between the movement vectors of the 4 images

    Args:
        positions (torch.Tensor): (x, y) position for the 4 images. Shape: (8)
        normalise (bool): Whether to normalise the distance

    Returns: Maximum distance (1,)
    """
    deltas = get_movement_vectors(positions)
    average_distance = torch.mean(torch.norm(deltas, dim=1)) if normalise else 1

    distances = []
    for delta1_idx, delta2_idx in torch.combinations(
        torch.arange(len(deltas)), r=2, with_replacement=False
    ):
        delta = deltas[delta1_idx] - deltas[delta2_idx]
        distances.append(torch.norm(delta))
    return torch.Tensor([torch.max(torch.stack(distances)) / average_distance])


def get_gradients(positions: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Returns the gradients of position changes between the 4 images

    Args:
        positions (torch.Tensor): (x, y) position for the 4 images. Shape: (8,)
        epsilon (float): Small value to prevent division by zero

    Returns: Gradients (3,)
    """
    deltas = get_movement_vectors(positions)

    gradients = []
    for i in range(0, len(deltas)):
        if deltas[i][0] < 0:
            gradients.append(deltas[i][1] / (deltas[i][0] - epsilon))
        else:
            gradients.append(deltas[i][1] / (deltas[i][0] + epsilon))

        if gradients[-1].isnan():
            print(f"{deltas[i][0]}, {deltas[i][1]}")
            print(f"NaN gradient detected: {gradients[-1]}")
            print(positions)
        if gradients[-1].isinf():
            print(f"Infinite gradient detected: {gradients[-1]}")
    return torch.stack(gradients)


def get_max_grad_diff(positions: torch.Tensor) -> torch.Tensor:
    """
    Returns the maximum difference in gradients between the 4 images

    Args:
        positions (torch.Tensor): (x, y) position for the 4 images. Shape: (8)

    Returns: Maximum difference in gradients (1,)
    """
    gradients = get_gradients(positions)

    grad_diffs = []
    for grad1, grad2 in torch.combinations(gradients, r=2, with_replacement=False):
        grad_diffs.append(abs(grad1 - grad2))

    # Calculate the maximum difference in gradients
    max_grad_diff = max(grad_diffs)
    return torch.Tensor([max_grad_diff])


def get_angles(positions: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Returns the angles of the lines between the positions of the 4 images

    Args:
        positions (torch.Tensor): (x, y) position for the 4 images. Shape: (8)

    Returns: Angles (3,)
    """
    deltas = get_movement_vectors(positions)

    angles = []
    for i in range(0, len(deltas)):
        if deltas[i][0] < 0:
            angle = torch.atan(deltas[i][1] / (deltas[i][0] - epsilon))
        else:
            angle = torch.atan(deltas[i][1] / (deltas[i][0] + epsilon))
        angles.append(angle * 180 / math.pi)

        if angles[-1].isnan():
            print(f"{deltas[i][0]}, {deltas[i][1]}")
            print(f"NaN angle detected: {angles[-1]}")
            print(positions)
        if angles[-1].isinf():
            print(f"Infinite angle detected: {angles[-1]}")
    return torch.stack(angles)


def get_max_ang_diff(positions: torch.Tensor) -> torch.Tensor:
    """
    Returns the maximum difference in angle between the 4 images

    Args:
        positions (torch.Tensor): (x, y) position for the 4 images. Shape: (8)

    Returns: Maximum difference in angle (1,)
    """
    angles = get_angles(positions)

    ang_diffs = []
    for ang1, ang2 in torch.combinations(angles, r=2, with_replacement=False):
        diff = abs(ang1 - ang2)
        if diff > 180:
            diff = 360 - diff
        ang_diffs.append(diff)

    return torch.Tensor([max(ang_diffs)])


def get_movement_vectors(positions: torch.Tensor) -> torch.Tensor:
    """
    Returns dx, dy movement vectors between the 4 images

    Args:
        positions (torch.Tensor): (x, y) position for the 4 images. Shape: (8,)

    Returns: Movement vectors (3, 2)
    """
    # Reshape to (4, (x, y))
    positions = positions.view(4, 2)

    deltas = []
    for i in range(1, len(positions)):
        deltas.append(positions[i] - positions[i - 1])

    return torch.stack(deltas)


def get_dataframe(real_movers_csv: str, bogus_movers_csv: str) -> pd.DataFrame:
    """
    Returns a DataFrame object grouped by the mover_id with columns file_name and label

    Args:
        real_movers_csv (str): Path to the csv file containing the real movers
        bogus_movers_csv (str): Path to the csv file containing the bogus movers
    """
    # Read csv
    real_movers = pd.read_csv(real_movers_csv)
    bogus_movers = pd.read_csv(bogus_movers_csv)

    # Ignore rows with missing data
    real_movers = real_movers.dropna(subset=[x_cord, y_cord, "file_name"])
    bogus_movers = bogus_movers.dropna(subset=[x_cord, y_cord, "file_name"])

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
) -> Tuple[torch.utils.data.TensorDataset, List[str]]:
    # Generate input, output pairs
    x_tensors = []
    y_hat_tensors = []
    mover_ids = []
    for mover_id, group_data in movers_agg:
        image_tensors = []
        # Ignore sequences that aren't 4 images long
        if len(group_data) != 4:
            # print(f"Skipping {mover_id} sequence with length: {len(group_data)}")
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
            y_hat_tensors.append(torch.Tensor([[group_data["label"].iloc[0]]]))
            mover_ids.append(mover_id)

    x = torch.concat(x_tensors)
    y_hat = torch.concat(y_hat_tensors)

    n_real_asteroids = y_hat.sum()
    n_bogus_asteroids = y_hat.shape[0] - n_real_asteroids
    print(
        f"Movers: {n_real_asteroids + n_bogus_asteroids}, Real asteroids: {n_real_asteroids}, Bogus asteroids: {n_bogus_asteroids}"
    )

    data_set = torch.utils.data.TensorDataset(x, y_hat)
    return data_set, mover_ids


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
