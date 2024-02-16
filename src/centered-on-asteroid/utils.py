import torch
import pandas as pd
import pandas.api.typing as pd_typing
from typing import List
from PIL import Image
import torchvision

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