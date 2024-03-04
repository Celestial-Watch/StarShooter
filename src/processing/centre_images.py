from PIL import Image
import numpy as np

def get_movers(csv):
    with open(csv, 'r') as file:
        csv_len = len(file.re)
        movers = np.zeros(csv_len, 2)
        big_images = np.zeros((csv_len, 4, 1024, 1024))
        small_images = np.zeros((csv_len, 4, 30, 30))
        for i, line in enumerate(file):
            line = line.strip()
            line = line.split(',')
            movers[i] = line[1]
            big_images[i] = fetch_big_images(line[0])
            small_images[i] = fetch_small_images(line[0])
    return movers, big_images, small_images

def fetch_small_images(image_name




def crop_image_around_pixel(image: np.ndarray, x: int, pixel_x: int, pixel_y: int, zero_padding: bool = False) -> Image:
    image = Image.fromarray(image)
    width, height = image.size
    half_x = x // 2
    left = max(0, pixel_x - half_x)
    upper = max(0, pixel_y - half_x)
    right = min(width, pixel_x + half_x)
    lower = min(height, pixel_y + half_x)

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

if __name__ == "__main__":
    image = np.array(Image.open("test.jpg"))
    cropped_image = crop_image_around_pixel(image, 100, 100, 100)
    cropped_image.show()
