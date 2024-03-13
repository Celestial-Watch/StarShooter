from PIL import Image
import numpy as np
import config
import pandas as pd
import pandas.api.typing as pd_typing
from matplotlib import pyplot as plt
import torchvision
import torch
from typing import List, Tuple
import os
from tqdm import tqdm


def images_to_fetch(csv_pos: str, csv_neg: str, images_per_mover: int = 4):

    df_pos = pd.read_csv(csv_pos)
    df_neg = pd.read_csv(csv_neg)

    movers_pos = pd.DataFrame({"mover_id": df_pos["mover_id"].unique()})
    movers_neg = pd.DataFrame({"mover_id": df_neg["mover_id"].unique()})

    movers_pos["label"] = 1
    movers_neg["label"] = 0
    print(movers_pos.shape)
    print(movers_neg.shape)

    movers = pd.concat([movers_pos, movers_neg])
    print(movers.shape)
    df = pd.concat([df_pos, df_neg])

    big_image_ids = np.empty((movers.shape[0], images_per_mover), dtype=object)
    small_image_ids = np.empty((movers.shape[0], images_per_mover), dtype=object)
    positions = np.empty((movers.shape[0], images_per_mover, 2), dtype=object)
    for i, mover in tqdm(enumerate(movers["mover_id"]), total=movers.shape[0]):
        big_images = np.array(df[df["mover_id"] == mover]["image_id"].to_list())
        small_images = df[df["mover_id"] == mover]["centred_image_id"].to_numpy()
        mover_positions = df[df["mover_id"] == mover][["X", "Y"]].to_numpy()

        big_image_ids[i] = big_images[:]
        small_image_ids[i] = small_images[:]
        positions[i] = mover_positions[:]

    return movers, small_image_ids, big_image_ids, positions


def fetch_small_images(
    small_image_ids: np.array,
    movers: pd.DataFrame,
    image_path: str,
    image_shape: Tuple = (30, 30),
    images_per_sequence: int = 4,
):
    # Determine the dimensions of the input array

    rows, cols = small_image_ids.shape
    print(f"{small_image_ids.shape}")

    # Initialize an empty array with the same dimensions to store the NumPy arrays of images
    images = np.empty((rows, cols), dtype=object)

    x_tensors = []
    y_hat_tensors = []
    movers_to_remove = []

    print("Importing Small Images")
    movers.reset_index(drop=True, inplace=True)
    for i, row in tqdm(movers.iterrows(), total=movers.shape[0]):
        image_tensors = []
        images = small_image_ids[i]
        mover_id = row["mover_id"]
        label = row["label"]

        # Ignore sequences that aren't 4 images long
        if len(images) != images_per_sequence:
            print(f"Skipping {mover_id} sequence with length: {len(images)}")
            continue

        for image_id in images:
            try:
                # Read image as PIL Image and convert to grayscale
                image_full_path = os.path.join(image_path, image_id)
                image = Image.open(image_full_path).convert("L")
            except FileNotFoundError:
                print(f"Image of {mover_id} not found: {image_full_path}")
                movers_to_remove.append(mover_id)
                break

            # Convert PIL Image to torch.Tensor
            transform = torchvision.transforms.ToTensor()
            image_tensor = transform(image)

            # Reshape image tensor to match the expected input shape
            try:
                image_tensor = image_tensor.view(1, 1, *image_shape)
                image_tensors.append(image_tensor)
            except RuntimeError as e:
                image_tensors = []
                movers_to_remove.append(mover_id)
                break
        else:
            # Loop finished without break
            # Concatenate over width dimension -> (1, 4, 30, 30)
            x_tensor = torch.cat(image_tensors, dim=1)
            x_tensors.append(x_tensor)
            y_hat_tensors.append(torch.tensor([[label]]))
    x = torch.concat(x_tensors)
    y_hat = torch.concat(y_hat_tensors)
    data_set = torch.utils.data.TensorDataset(x, y_hat)
    # print(f"Data shape: {x.shape}")
    # print(f"Label shape: {y_hat.shape}")
    # print(f"data_set shape: {data_set.shape}")
    print("Full sets not found for {} movers".format(len(movers_to_remove)))
    movers = movers[~movers["mover_id"].isin(movers_to_remove)]

    return data_set, movers


def fetch_cropped_images(
    big_image_ids: np.array,
    movers: pd.DataFrame,
    image_path: str,
    positions: np.array,
    images_per_sequence: int = 4,
    crop_size: int = 100,
):
    # Determine the dimensions of the input array
    rows, cols = big_image_ids.shape

    # Initialize an empty array with the same dimensions to store the NumPy arrays of images
    images = np.empty((rows, cols), dtype=object)
    pixel_coords = np.empty((rows, cols, 2), dtype=int)

    # Iterate through the 2D array of image IDs
    x_tensors = []
    y_hat_tensors = []
    movers_to_remove = []
    images_not_found = []
    print("Importing and Cropping Large Images")
    movers.reset_index(drop=True, inplace=True)

    for i, row in tqdm(movers.iterrows(), total=movers.shape[0]):
        image_tensors = []
        images = big_image_ids[i]
        mover_id = row["mover_id"]
        label = row["label"]

        # Ignore sequences that aren't 4 images long
        if len(images) != images_per_sequence:
            print(f"Skipping {mover_id} sequence with length: {len(images)}")
            continue

        for j, image_id in enumerate(images):
            try:
                # Read image as PIL Image and convert to grayscale
                image_full_path = f"{image_path}{image_id}.png"
                image = Image.open(image_full_path).convert("L")
            except Exception as e:
                movers_to_remove.append(mover_id)
                images_not_found.append(image_id)
                break

            # Get the pixel coordinates for the mover and image with a combination key
            pixel_coords = positions[i, j]

            # Crop the image around the pixel coordinates
            cropped_image = crop_image_around_pixel(
                image, crop_size, round(pixel_coords[0]), round(pixel_coords[1])
            )

            # Convert PIL Image to torch.Tensor
            transform = torchvision.transforms.ToTensor()
            image_tensor = transform(cropped_image)

            # Reshape image tensor to match the expected input shape
            try:
                image_tensor = image_tensor.view(1, 1, crop_size, crop_size)
                image_tensors.append(image_tensor)
            except RuntimeError as e:
                image_tensors = []
                print(e)
                movers_to_remove.append(mover_id)
                break
        else:
            # Loop finished without break
            # Concatenate over width dimension -> (1, 4, 30, 30)
            x_tensor = torch.cat(image_tensors, dim=1)
            x_tensors.append(x_tensor)
            y_hat_tensors.append(torch.tensor([[label]]))

    x = torch.concat(x_tensors)
    y_hat = torch.concat(y_hat_tensors)
    data_set = torch.utils.data.TensorDataset(x, y_hat)
    print("Full sets not found for {} movers".format(len(movers_to_remove)))
    movers = movers[~movers["mover_id"].isin(movers_to_remove)]
    print(images_not_found)
    images_not_found_set = set(images_not_found)
    # write the images not found to csv
    with open(
        os.path.join(config.PROCESSING_PATH, "data/images_not_found.csv"), "w"
    ) as f:
        for image in images_not_found:
            f.write(f"{image}\n")
    all_whole_images = [
        image.replace(".png", "")
        for image in os.listdir(
            os.path.join(config.PROCESSING_PATH, "data/alistair/images")
        )
    ]
    all_whole_images_set = set(all_whole_images)
    print(f"Set of length of set not found: {len(images_not_found_set)}")
    print(f"Set of whole images: {len(all_whole_images_set)}")
    print(
        f"Intersection between images not found and whole images: {len(images_not_found_set.intersection(all_whole_images_set))}"
    )
    return data_set, pixel_coords, movers


def crop_image_around_pixel(
    image: Image, crop_size: int, pixel_x: int, pixel_y: int
) -> Image:
    width, height = image.size
    half_crop = crop_size // 2
    left = max(0, pixel_x - half_crop)
    upper = max(0, pixel_y - half_crop)
    right = min(width, pixel_x + half_crop)
    lower = min(height, pixel_y + half_crop)

    left_padding = right_padding = upper_padding = lower_padding = 0

    if left == 0:
        left_padding = half_crop - pixel_x
        left = 0
    if upper == 0:
        upper_padding = half_crop - pixel_y
        upper = 0
    if right == width:
        right_padding = half_crop - (width - pixel_x)
        right = width
    if lower == height:
        lower_padding = half_crop - (height - pixel_y)
        lower = height

    cropped_image = image.crop((left, upper, right, lower))

    if left_padding > 0 or upper_padding > 0 or right_padding > 0 or lower_padding > 0:
        padded_image = Image.new(image.mode, (crop_size, crop_size))
        padded_image.paste(cropped_image, (left_padding, upper_padding))
        cropped_image = padded_image

    return cropped_image


def get_datasets(crop_size: int, images_per_sequence=4):

    movers, small_image_ids, big_image_ids, mover_positions = images_to_fetch(
        config.POS_MOVER_PATH, config.NEG_MOVER_PATH
    )

    cropped_image_set, _, movers = fetch_cropped_images(
        big_image_ids,
        movers,
        config.BIG_IMAGE_PATH,
        mover_positions,
        images_per_sequence,
        crop_size,
    )

    small_image_set, _ = fetch_small_images(
        small_image_ids, movers, config.SMALL_IMAGE_PATH
    )

    return small_image_set, cropped_image_set


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


if __name__ == "__main__":
    get_datasets(100)
