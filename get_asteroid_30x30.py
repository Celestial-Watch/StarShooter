from tqdm import tqdm
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
import os
from argparse import ArgumentParser

base = "https://totas.cosmos.esa.int/"
cookie_token = "PHPSESSID=363c79f1832ff4318156867127ab18ac"


# This function makes the request to the server and downloads the images
# only if four a present
def get_four_30x30(url: str, mover_id: str):
    r = requests.get(url, headers={"Cookie": cookie_token})
    soup = BeautifulSoup(r.content, "html.parser")
    contents = soup.findAll("tr")
    for i, content in enumerate(contents):
        if content.find("td").find("br"):
            break
    image_list = [img["src"] for img in contents[i + 1].findAll("img")[1:]]
    print(image_list)
    if len(image_list) == 4:
        image_names = [link.split("/")[-1] for link in image_list]
        for i, image in enumerate(image_list):
            os.system(f"wget {base + image} -O 30x30_images/{image_names[i]}")
            time.sleep(1)

        with open(mover_image_csv, "a") as f:
            for image in image_names:
                f.write(f"{mover_id},{image}\n")
    else:
        print(f"Skipping {mover_id}")


# This function is used to save the progress of the save
# taking an index as input
def save_progress(index: int):
    with open(tracking_file, "w") as f:
        f.write(str(index))


# This function will return the index of the last mover download
def load_progress():
    if os.path.exists(tracking_file):
        with open(tracking_file, "r") as f:
            return int(f.read())
    else:
        return 0


if __name__ == "__main__":
    parse = ArgumentParser()
    parse.add_argument("--mover_file", type=str, default="movers.csv")
    parse.add_argument("--progress_file", type=str, default="scraper_progress.txt")

    args = parse.parse_args()

    mover_csv = pd.read_csv(args.mover_file)
    tracking_file = args.progress_file
    mover_image_csv = f"{args.mover_file}_images_lookup.csv"

    current_index = load_progress()
    for i in tqdm(range(current_index, len(mover_csv))):
        get_four_30x30(base + mover_csv["Link"][i], mover_csv["Name"][i])
        save_progress(i)
        time.sleep(1)