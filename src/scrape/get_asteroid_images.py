from tqdm import tqdm
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
import os

base = "https://totas.cosmos.esa.int/"
metadata = pd.read_csv("metadata.csv")


def get_image(url, id_):
    r = requests.get(
        url, headers={"Cookie": "PHPSESSID=aa35e90d2f77d812aa8cc165314d3ae8"}
    )
    soup = BeautifulSoup(r.content, "html.parser")
    img = soup.findAll("img")[-1]
    img_url = base + img["src"]
    os.system(f"wget {img_url} -O images/{id_}.png")


for i in tqdm(range(len(metadata))):
    get_image(base + metadata["Link"][i], metadata["id"][i])
    time.sleep(0.7)
