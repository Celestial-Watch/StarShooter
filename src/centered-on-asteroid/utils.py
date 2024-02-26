import torch
import pandas as pd
import pandas.api.typing as pd_typing
from typing import List
from PIL import Image
import torchvision
from typing import Tuple
import math


def get_position_tensor(
    movers_agg: pd_typing.DataFrameGroupBy,
) -> torch.Tensor:
    movers_positions = []
    for mover_id, group_data in movers_agg:
        # Ignore sequences that aren't 4 images long
        if len(group_data) != 4:
            # print(f"Skipping {mover_id} sequence with length: {len(group_data)}")
            continue

        mover_positions = []
        for _, row in group_data.iterrows():
            if math.isnan(row["pos_X"]) or math.isnan(row["pos_Y"]):
                print(f"Missing position data for {mover_id}")
                break
            mover_positions.append(row["pos_X"])
            mover_positions.append(row["pos_Y"])
        movers_positions.append(torch.Tensor(mover_positions))
    return torch.stack(movers_positions)


def get_engineered_features(movers_positions: torch.Tensor) -> torch.Tensor:
    get_max_grad_diffs = []
    for mover_positions in movers_positions:
        get_max_grad_diffs.append(torch.Tensor([get_max_grad_diff(mover_positions)]))
    return torch.stack(get_max_grad_diffs)


def get_max_grad_diff(positions: torch.Tensor) -> torch.Tensor:
    """
    Returns the maximum difference in gradients between the 4 images

    Args:
        positions (torch.Tensor): A tensor of shape (8)
    """
    # Reshape to (4, 2)
    positions = positions.view(4, 2)

    deltas = []
    for i in range(1, len(positions)):
        deltas.append(positions[i] - positions[i - 1])

    gradients = []
    epsilon = 1e-6
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
    gradients = torch.stack(gradients)

    grad_diffs = []
    for grad1, grad2 in torch.combinations(gradients, r=2, with_replacement=False):
        grad_diffs.append(abs(grad1 - grad2))

    # Calculate the maximum difference in gradients
    max_grad_diff = max(grad_diffs)
    return torch.Tensor([max_grad_diff])


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
    real_movers = real_movers.dropna(subset=["pos_X", "pos_Y", "file_name"])
    bogus_movers = bogus_movers.dropna(subset=["pos_X", "pos_Y", "file_name"])

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
            y_hat_tensors.append(torch.tensor([[group_data["label"].iloc[0]]]))
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
