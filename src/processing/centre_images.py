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


def images_to_fetch(
        csv_pos: str, 
        csv_neg: str,
        images_per_mover:int = 4
        ):
    
    df_pos = pd.read_csv(csv_pos)
    df_neg = pd.read_csv(csv_neg)
    
    movers_pos = df_pos['mover_id'].unique()
    movers_neg =  df_neg['mover_id'].unique()
    
    movers_pos['label'] = 1
    movers_neg['label'] = 0

    movers = pd.concat([movers_pos, movers_neg])

    big_image_ids = np.empty((movers.shape[0], images_per_mover), dtype=object)
    small_image_ids = np.empty((movers.shape[0], images_per_mover), dtype=object)
    for label, df in enumerate([df_neg, df_pos]):
        for i, mover in enumerate(df['mover_id']):
            big_images = np.array(df[df['mover_id'] == mover]['image_id'].tolist())
            small_images = np.array(df[df['mover_id'] == mover]['centered_image_id'].tolist())
            
            big_image_ids[i] = big_images[:]
            small_image_ids[i] = small_images[:]

    return movers, small_image_ids, big_image_ids

def fetch_small_images(
        small_image_ids: np.array, 
        movers: pd.DataFrame,
        image_path: str, 
        image_shape: Tuple = (30, 30),
        images_per_sequence: int = 3
        ):
    # Determine the dimensions of the input array

    rows, cols = small_image_ids.shape
    
    # Initialize an empty array with the same dimensions to store the NumPy arrays of images
    images = np.empty((rows, cols), dtype=object)

    
    x_tensors = []
    y_hat_tensors = []
    movers_to_remove = []
    
    for i, row in movers.iterrows():
        image_tensors = []
        images = small_image_ids[i]
        mover_id = row['mover_id']
        label = row['label']

        # Ignore sequences that aren't 4 images long
        if len(images) != images_per_sequence:
            print(f"Skipping {mover_id} sequence with length: {len(images)}")
            continue

        for image_id in images:
            print(f"Fetching {image_id}")
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
            y_hat_tensors.append(torch.tensor([label]))

    x = torch.concat(x_tensors)
    y_hat = torch.concat(y_hat_tensors)
    data_set = torch.utils.data.TensorDataset(x, y_hat)
    return data_set, movers_to_remove

def fetch_cropped_images(
        big_image_ids: np.array, 
        movers: np.array,
        movers_to_remove: np.array, 
        image_path: str, 
        metadata_path: str,
        images_per_sequence: int = 3,
        crop_size: int = 100
        ):
    # Determine the dimensions of the input array
    rows, cols = big_image_ids.shape
    
    # Initialize an empty array with the same dimensions to store the NumPy arrays of images
    images = np.empty((rows, cols), dtype=object)
    pixel_coords = np.empty((rows, cols, 2), dtype=int)
    
    # Load the metadata file as a dataframe
    metadata = pd.read_csv(metadata_path)

    # Iterate through the 2D array of image IDs
    x_tensors = []
    y_hat_tensors = []
    mover_ids = []
    for i, row in movers.iterrows():
        image_tensors = []
        images = big_image_ids[i]
        mover_id = row['mover_id']
        label = row['label']
        if mover_id in movers_to_remove:
            continue

        # Ignore sequences that aren't 4 images long
        if len(images) != images_per_sequence:
            print(f"Skipping {mover_id} sequence with length: {len(images)}")
            continue

        for image_id in images:
            print(image_id)
            try:
                # Read image as PIL Image and convert to grayscale
                image_full_path = f"{image_path}{image_id}.png"
                print(image_full_path)
                image = Image.open(image_full_path).convert("L")
            except FileNotFoundError:
                print(f"Image of {mover_id} not found: {image_full_path}")
                break

            # Get the pixel coordinates for the mover and image with a combination key
            pixel_coords = metadata.loc[(metadata['mover_id'] == mover_id) & (metadata['image_id'] == big_image_ids[i][j])][['X', 'Y']].values[0]


            # Crop the image around the pixel coordinates
            cropped_image = crop_image_around_pixel(image, crop_size, pixel_coords[i][j][0], pixel_coords[i][j][1])
            
            # Convert PIL Image to torch.Tensor
            transform = torchvision.transforms.ToTensor()
            image_tensor = transform(cropped_image)
        
            # Reshape image tensor to match the expected input shape
            try:
                image_tensor = image_tensor.view(1, 1, crop_size, crop_size)
                image_tensors.append(image_tensor)
            except RuntimeError as e:
                image_tensors = []
                break
        else:
            # Loop finished without break
            # Concatenate over width dimension -> (1, 4, 30, 30)
            x_tensor = torch.cat(image_tensors, dim=1)
            x_tensors.append(x_tensor)
            y_hat_tensors.append(torch.tensor([label]))
            mover_ids.append(mover_id)

    x = torch.concat(x_tensors)
    y_hat = torch.concat(y_hat_tensors)
    data_set = torch.utils.data.TensorDataset(x, y_hat)
    return data_set, pixel_coords



def crop_image_around_pixel(
        image: np.ndarray, 
        crop_size: int, 
        pixel_x: int, 
        pixel_y: int
        ) -> Image:
    image = Image.fromarray(image)
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

def crop_images(images, positions, crop_size):
    cropped_images = np.empty((images.shape[0], images.shape[1], crop_size, crop_size))
    for i in range(len(images)):
        for j in range(len(images[i])):
            image = images[i][j]
            position = positions[i][j]
            pixel_x = round(position[0])
            pixel_y = round(position[1])
            cropped_image = crop_image_around_pixel(image, crop_size, pixel_x, pixel_y)
            cropped_images[i][j] = cropped_image
    return cropped_images

def test_crop_image_around_pixel():
    image = np.zeros((100, 100))
    image[50, 50] = 255
    cropped_image = crop_image_around_pixel(image, 30, 50, 50)
    assert cropped_image.size == (30, 30)
    
    plt.gray()
    plt.imshow(cropped_image)
    plt.show()

    image = np.zeros((100, 100))
    image[10, 10] = 255
    cropped_image = crop_image_around_pixel(image, 30, 10, 10)
    assert cropped_image.size == (30, 30)

    plt.gray()
    plt.imshow(cropped_image)
    plt.show()

    image = np.zeros((100, 100))
    image[90, 90] = 255
    cropped_image = crop_image_around_pixel(image, 30, 90, 90)
    assert cropped_image.size == (30, 30)

    plt.gray()
    plt.imshow(cropped_image)
    plt.show()

    image = np.zeros((100, 100))
    image[0, 0] = 255
    cropped_image = crop_image_around_pixel(image, 30, 0, 0)
    assert cropped_image.size == (30, 30)

    plt.gray()
    plt.imshow(cropped_image)
    plt.show()

    image = np.zeros((100, 100))
    image[0, 0] = 255
    cropped_image = crop_image_around_pixel(image, 30, 100, 100)
    assert cropped_image.size == (30, 30)

    plt.gray()
    plt.imshow(cropped_image)
    plt.show()

def get_datasets(
        crop_size: int
        ):

    images_per_sequence = 4
    movers, small_image_ids, big_image_ids = images_to_fetch(config.POS_MOVER_PATH, config.NEG_MOVER_PATH)
    movers = get_mover_labels(movers, config.MOVERS_PATH)
    print(len(movers))

    small_image_set, movers_to_remove = fetch_small_images(small_image_ids, movers, config.SMALL_IMAGE_PATH)
    cropped_image_set, mover_positions = fetch_cropped_images(big_image_ids, movers, movers_to_remove, config.BIG_IMAGE_PATH, config.POSITION_PATH, images_per_sequence, crop_size,)

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

get_datasets(100)