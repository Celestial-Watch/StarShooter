from tqdm import tqdm
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
import os


def get_mover_to_image_mapping(mover_id):
    base = "https://totas.cosmos.esa.int/mover.php?id="
    url = base + str(mover_id)
    request = requests.get(url)
    soup = BeautifulSoup(request.text, "html.parser")
    images = soup.findAll("img")
    images = [
        image["src"]
        for image in images
        if "b" in image["src"] and ".png" in image["src"]
    ]
    images = list(set(images))
    images = [image.split("/")[-1] for image in images]

    return images


if __name__ == "__main__":
    movers = pd.read_csv("movers_2021-2024_cleaned.txt")
    movers_mapping_file = "mover_to_image_mapping.csv"
    for mover_id in tqdm(movers["totas_id"].unique()):
        images = get_mover_to_image_mapping(mover_id)
        if len(images) > 0:
            with open(movers_mapping_file, "a") as f:
                for image in images:
                    f.write(f"{mover_id},{image}\n")
