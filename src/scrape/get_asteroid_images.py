from tqdm import tqdm
import pandas as pd
import time
from config import BASE, IMAGE_FOLDER
from utils import download_whole_image

metadata = pd.read_csv("metadata.csv")

for i in tqdm(range(len(metadata))):
    download_whole_image(
        BASE + metadata["Link"][i], metadata["id"][i], f"{IMAGE_FOLDER}/whole_images"
    )
    time.sleep(1)
