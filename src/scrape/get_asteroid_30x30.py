from tqdm import tqdm
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
import os
from argparse import ArgumentParser
from utils import save_progress, load_progress
from config import BASE, SESSION_ID


# This function makes the request to the server and downloads the images
# only if four a present
def get_four_30x30(
    url: str,
    mover_id: str,
    mover_image_csv: str,
    base: str = BASE,
    cookie_token: str = SESSION_ID,
):
    r = requests.get(url, headers={"Cookie": cookie_token})
    soup = BeautifulSoup(r.content, "html.parser")
    contents = soup.findAll("tr")
    for i, content in enumerate(contents):
        if content.find("td").find("br"):
            break
    image_list = [img["src"] for img in contents[i + 1].findAll("img")[1:]]
    if len(image_list) == 4:
        image_names = [link.split("/")[-1] for link in image_list]
        for i, image in enumerate(image_list):
            os.system(f"wget {base + image} -O 30x30_images/{image_names[i]}")
            time.sleep(1)
            # For this command to work ensure that you have sign in the correct gcloud profile and project
            os.system(
                f"gsutil -m cp 30x30_images/{image_names[i]} gs://mlp-asteroid-data/30x30_images"
            )

        with open(mover_image_csv, "a") as f:
            for image in image_names:
                f.write(f"{mover_id},{image}\n")
    else:
        print(f"Skipping {mover_id}")


if __name__ == "__main__":
    parse = ArgumentParser()
    parse.add_argument("--mover_file", type=str, default="movers.csv")
    parse.add_argument("--progress_file", type=str, default="scraper_progress.txt")

    args = parse.parse_args()

    mover_csv = pd.read_csv(args.mover_file)
    tracking_file = args.progress_file
    mover_image_csv_path = args.mover_file.split(".")[0]
    mover_image_csv = f"{mover_image_csv_path}_images_lookup.csv"

    upload_bucket = "gs://mlp-asteroid-data/csv/"
    upload_checker = 0

    current_index = load_progress(tracking_file)
    for i in tqdm(range(current_index, len(mover_csv))):
        get_four_30x30(
            BASE + mover_csv["Link"][i], mover_csv["Name"][i], mover_image_csv
        )
        save_progress(tracking_file, i)
        time.sleep(1)
        upload_checker += 1

        if upload_checker == 10:
            os.system(f"gsutil -m cp {mover_image_csv} {upload_bucket}")
            upload_checker = 0
            time.sleep(2)
