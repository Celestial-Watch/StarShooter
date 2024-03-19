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


def images_to_fetch(csv: str, images_per_mover: int = 4):

    df = pd.read_csv(csv)

    movers = pd.DataFrame(
        {
            "mover_id": df["mover_id"].unique(),
            "label": df.groupby("mover_id")["label"].first(),
        }
    ).reset_index(drop=True)

    print(movers.shape)

    big_image_ids = np.empty((movers.shape[0], images_per_mover), dtype=object)
    small_image_ids = np.empty((movers.shape[0], images_per_mover), dtype=object)
    positions = np.empty((movers.shape[0], images_per_mover, 2), dtype=object)
    big_image_df = pd.DataFrame()
    small_image_df = pd.DataFrame()

    movers_list = np.empty(movers.shape[0], dtype=str)

    for i, mover in tqdm(enumerate(movers["mover_id"]), total=movers.shape[0]):
        big_images = np.array(df[df["mover_id"] == mover]["totas_id"].to_list())
        small_images = df[df["mover_id"] == mover]["file_name"].to_numpy()
        mover_positions = df[df["mover_id"] == mover][["X", "Y"]].to_numpy()

        big_image_ids[i] = big_images[:]
        small_image_ids[i] = small_images[:]
        positions[i] = mover_positions[:]
        movers_list[i] = mover

    movers_safe = movers.copy()

    big_image_df = big_image_df.assign(
        mover_id=movers["mover_id"].tolist(),
        label=movers["label"].tolist(),
        image1=big_image_ids[:, 0].tolist(),
        image2=big_image_ids[:, 1].tolist(),
        image3=big_image_ids[:, 2].tolist(),
        image4=big_image_ids[:, 3].tolist(),
        position1=positions[:, 0].tolist(),
        position2=positions[:, 1].tolist(),
        position3=positions[:, 2].tolist(),
        position4=positions[:, 3].tolist(),
    )

    small_image_df = small_image_df.assign(
        mover_id=movers["mover_id"].tolist(),
        label=movers["label"].tolist(),
        image1=small_image_ids[:, 0].tolist(),
        image2=small_image_ids[:, 1].tolist(),
        image3=small_image_ids[:, 2].tolist(),
        image4=small_image_ids[:, 3].tolist(),
    )

    return movers, small_image_df, big_image_df


def fetch_small_images(
    small_image_df: pd.DataFrame,
    movers: pd.DataFrame,
    image_path: str,
    image_shape: Tuple = (30, 30),
    images_per_sequence: int = 4,
    image_name_extension: str = "",
):
    # Initialize an empty array with the same dimensions to store the NumPy arrays of images
    images = np.empty((len(movers), images_per_sequence), dtype=object)

    x_tensors = []
    y_hat_tensors = []
    movers_to_remove = []

    print("Importing Small Images")
    for _, row in tqdm(small_image_df.iterrows(), total=small_image_df.shape[0]):
        image_tensors = []
        images = row[["image1", "image2", "image3", "image4"]].tolist()
        mover_id = row["mover_id"]
        label = row["label"]

        # if mover_id not in movers["mover_id"].to_list():
        #     continue

        # Ignore sequences that aren't 4 images long
        if len(images) != images_per_sequence:
            print(f"Skipping {mover_id} sequence with length: {len(images)}")
            continue

        for image_id in images:
            try:
                # Read image as PIL Image and convert to grayscale
                image_full_path = os.path.join(
                    image_path, str(image_id) + image_name_extension
                )
                image = Image.open(image_full_path).convert("L")
            except FileNotFoundError:
                print(f"Image of {mover_id} not found: {image_full_path}")
                movers_to_remove.append(mover_id)
                break

            # Convert PIL Image to torch.Tensor
            transform = torchvision.transforms.ToTensor()
            image_tensor = transform(image)

            # Normalise the image
            image_tensor = torch.nn.functional.normalize(image_tensor, 1)

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
    big_image_df: pd.DataFrame,
    movers: pd.DataFrame,
    image_path: str,
    images_per_sequence: int = 4,
    crop_size: int = 100,
):
    # Determine the dimensions of the input array

    # Initialize an empty array with the same dimensions to store the NumPy arrays of images
    images = np.empty((len(movers), images_per_sequence), dtype=object)
    pixel_coords = np.empty((len(movers), images_per_sequence, 2), dtype=object)

    # Determine whether cropped files have been created before
    path = os.path.join(
        config.PROCESSING_PATH, f"data/alistair/cropped_images_{crop_size}"
    )

    try:
        os.mkdir(path)
    except OSError as e:
        return fetch_small_images(
            big_image_df,
            movers,
            path,
            (crop_size, crop_size),
            images_per_sequence,
            "_" + str(crop_size) + ".png",
        )

    # Iterate through the 2D array of image IDs
    x_tensors = []
    y_hat_tensors = []
    movers_to_remove = []
    images_not_found = []
    print("Importing and Cropping Large Images")
    for i, row in tqdm(big_image_df.iterrows(), total=big_image_df.shape[0]):
        image_tensors = []
        mover_id = row["mover_id"]
        label = row["label"]

        images = row[["image1", "image2", "image3", "image4"]].tolist()
        positions = row[["position1", "position2", "position3", "position4"]].tolist()

        # if mover_id not in movers["mover_id"].to_list():
        #     continue

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
            pixel_coords = positions[j]

            # Crop the image around the pixel coordinates
            cropped_image = crop_image_around_pixel(
                image, crop_size, round(pixel_coords[0]), round(pixel_coords[1])
            )

            # Convert PIL Image to torch.Tensor
            transform = torchvision.transforms.ToTensor()
            image_tensor = transform(cropped_image)

            # Save image to ./data/alistair/cropped_images_100
            cropped_image.save(
                os.path.join(
                    path,
                    f"{image_id}_{crop_size}.png",
                )
            )

            # Normalise the image
            image_tensor = torch.nn.functional.normalize(
                image_tensor,
                1,
            )

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
    return data_set, movers


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

    movers, small_image_df, big_image_df = images_to_fetch(config.ALL_MOVERS_PATH)

    cropped_image_set, movers = fetch_cropped_images(
        big_image_df,
        movers,
        config.BIG_IMAGE_PATH,
        images_per_sequence,
        crop_size,
    )

    small_image_set, _ = fetch_small_images(
        small_image_df, movers, config.SMALL_IMAGE_PATH
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
