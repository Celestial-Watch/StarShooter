from PIL import Image
import numpy as np
import config
import pandas as pd
from matplotlib import pyplot as plt
import torchvision
import torch


def get_mover_labels(movers_list, csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Filter the DataFrame to only include rows where the mover_id is in the movers_list
    filtered_df = df[df['mover_id'].isin(movers_list)]
    
    # Initialize an empty array to store the results
    # Assuming the 'movers_list' is a list of mover IDs you're interested in
    # The length of movers array is now the length of the filtered DataFrame
    movers = np.empty((len(filtered_df), 2))
    
    # Iterate through the filtered DataFrame
    for i, (index, row) in enumerate(filtered_df.iterrows()):
        # The first column is assumed to be 'mover_id'
        movers[i][0] = row['mover_id']
        # Check the third column for the label, assuming it's named appropriately in the DataFrame
        if row['label'] == "background noise":  # Replace 'label_column_name' with the actual name
            movers[i][1] = 0
        else:
            movers[i][1] = 1
            
    return movers

def images_to_fetch(csv, images_per_mover=4):
    print(csv)
    df = pd.read_csv(csv)
    mover_counts = df['mover_id'].value_counts()
    movers = mover_counts[mover_counts == images_per_mover].index.tolist()

    big_image_ids = []
    small_image_ids = []
    for mover in movers:
        big_images = df[df['mover_id'] == mover]['image_id'].tolist()
        small_images = df[df['mover_id'] == mover]['centered_image_id'].tolist()
        
        big_image_ids.append(big_images)
        small_image_ids.append(small_images)

    return movers, np.array(small_image_ids, dtype=str), np.array(big_image_ids, dtype=str)


def fetch_small_images(small_image_ids, image_path, image_shape=(30, 30)):
    # Determine the dimensions of the input array
    rows, cols = small_image_ids.shape
    
    
    # Initialize an empty array with the same dimensions to store the NumPy arrays of images
    images = np.empty((rows, cols), dtype=object)
    
    # Iterate through the 2D array of image IDs
    for i in range(rows):
        image_tensors = []

        for j in range(cols):
            # Construct the full path for each image
            image_file = f"{image_path}/{small_image_ids[i][j]}"
            
            # Load the image using PIL, convert to grayscale, and then convert to a NumPy array
            try:
                img = Image.open(image_file).convert('L')
            except FileNotFoundError:
                print(f"Image {small_image_ids[i][j]} not found in {image_path}.")
                images[i][j] = None  # Use None or a placeholder image if the image is not found

            # Convert PIL Image to torch.Tensor
            transform = torchvision.transforms.ToTensor()
            image_tensor = transform(img)

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
    
    return images


def fetch_big_images(big_image_ids, movers, image_path, metadata_path):
    # Determine the dimensions of the input array
    rows, cols = big_image_ids.shape
    
    # Initialize an empty array with the same dimensions to store the NumPy arrays of images
    images = np.empty((rows, cols), dtype=object)
    pixel_coords = np.empty((rows, cols, 2), dtype=int)
    
    # Load the metadata file as a dataframe
    metadata = pd.read_csv(metadata_path)

    # Iterate through the 2D array of image IDs
    for i in range(rows):
        for j in range(cols):
            # Construct the full path for each image
            image_file = f"{image_path}/{big_image_ids[i][j]}.png"
            
            # Load the image using PIL, convert to grayscale, and then convert to a NumPy array
            try:
                img = Image.open(image_file).convert('L')
                images[i][j] = np.array(img)
            except FileNotFoundError:
                print(f"Image {big_image_ids[i][j]} not found in {image_path}.")
                images[i][j] = None  # Use None or a placeholder image if the image is not found
            
            mover = movers[i][0]

            # Get the pixel coordinates for the mover and image with a combination key
            pixel_coords[i][j] = metadata.loc[(metadata['mover_id'] == mover) & (metadata['image_id'] == big_image_ids[i][j])][['X', 'Y']].values[0]
    
    return images, pixel_coords



def crop_image_around_pixel(image: np.ndarray, x: int, pixel_x: int, pixel_y: int, zero_padding: bool = False) -> Image:
    image = Image.fromarray(image)
    width, height = image.size
    half_x = x // 2
    left = max(0, pixel_x - half_x)
    upper = max(0, pixel_y - half_x)
    right = min(width, pixel_x + half_x)
    lower = min(height, pixel_y + half_x)

    left_padding = right_padding = upper_padding = lower_padding = 0

    if left == 0:
        left_padding = half_x - pixel_x
        left = 0
    if upper == 0:
        upper_padding = half_x - pixel_y
        upper = 0
    if right == width:
        right_padding = half_x - (width - pixel_x)
        right = width
    if lower == height:
        lower_padding = half_x - (height - pixel_y)
        lower = height

    cropped_image = image.crop((left, upper, right, lower))

    if left_padding > 0 or upper_padding > 0 or right_padding > 0 or lower_padding > 0:
        padded_image = Image.new(image.mode, (x, x))
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

def get_datasets(crop_size):

    movers, small_image_ids, big_image_ids = images_to_fetch(config.POSITION_PATH)
    movers = get_mover_labels(movers, config.MOVERS_PATH)
    print(len(movers))

    small_image_set = fetch_small_images(small_image_ids, config.SMALL_IMAGE_PATH)
    big_image_set, mover_positions = fetch_big_images(big_image_ids, movers, config.BIG_IMAGE_PATH)
    #cropped_image_set = crop_images(big_image_set, mover_positions, crop_size)

def get_loaders(image_shape, batch_size):
    small_image_set, cropped_image_set, movers = get_datasets(image_shape[0])

test_crop_image_around_pixel()