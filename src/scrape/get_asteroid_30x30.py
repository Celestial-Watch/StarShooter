from tqdm import tqdm
import pandas as pd
import time
import os
from argparse import ArgumentParser
from utils import save_progress, load_progress, get_n_centered_on_asteroid
from config import BASE, IMAGE_FOLDER

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
        get_n_centered_on_asteroid(
            BASE + mover_csv["Link"][i],
            mover_csv["Name"][i],
            mover_image_csv,
            f"{IMAGE_FOLDER}/30x30_images",
        )
        save_progress(tracking_file, i)
        time.sleep(1)
        upload_checker += 1

        if upload_checker == 10:
            os.system(f"gsutil -m cp {mover_image_csv} {upload_bucket}")
            upload_checker = 0
            time.sleep(2)
