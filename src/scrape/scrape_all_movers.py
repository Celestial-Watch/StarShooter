import time
from tqdm import tqdm
import os
from argparse import ArgumentParser
from utils import save_progress, load_progress, download_mover_data
from config import (
    META_DATA_COLUM_NAMES,
    POSITION_TABLE_COLUMN_NAMES,
    CSV_FOLDER,
    IMAGE_FOLDER,
    BASE,
    MOVER_TABLE_COLUMN_NAMES,
)
import numpy as np

TOTAL_MOVERS = 400_000
bad_request_output = "<p>\n nice try.\n <br/>\n logged.\n <br/>\n bye.\n</p>\n"

if __name__ == "__main__":
    image_csv = "image.csv"
    image_csv_path = f"{CSV_FOLDER}/all_movers/{image_csv}"
    if not os.path.exists(image_csv_path):
        if not os.path.exists(f"{CSV_FOLDER}/all_movers"):
            os.makedirs(f"{CSV_FOLDER}/all_movers")

        with open(image_csv_path, "w") as f:
            f.write(",".join(META_DATA_COLUM_NAMES) + "\n")

    position_csv = "position.csv"
    position_csv_path = f"{CSV_FOLDER}/all_movers/{position_csv}"
    if not os.path.exists(position_csv_path):
        if not os.path.exists(f"{CSV_FOLDER}/all_movers"):
            os.makedirs(f"{CSV_FOLDER}/all_movers")

        with open(position_csv_path, "w") as f:
            f.write(",".join(POSITION_TABLE_COLUMN_NAMES) + "\n")

    mover_csv = "mover.csv"
    mover_csv_path = f"{CSV_FOLDER}/all_movers/{mover_csv}"
    if not os.path.exists(mover_csv_path):
        if not os.path.exists(f"{CSV_FOLDER}/all_movers"):
            os.makedirs(f"{CSV_FOLDER}/all_movers")

        with open(mover_csv_path, "w") as f:
            f.write(",".join(MOVER_TABLE_COLUMN_NAMES) + "\n")

    centered_on_asteroid_image_folder_rel = "centered_on_asteroid"
    centered_image_folder = f"{IMAGE_FOLDER}/{centered_on_asteroid_image_folder_rel}"
    if not os.path.exists(centered_image_folder):
        os.makedirs(centered_image_folder)

    whole_image_folder_rel = "whole_images"
    whole_image_folder = f"{IMAGE_FOLDER}/{whole_image_folder_rel}"
    if not os.path.exists(whole_image_folder):
        os.makedirs(whole_image_folder)

    # GS definitions
    upload_bucket_base = "gs://mlp-asteroid-data"
    upload_bucket_images = f"{upload_bucket_base}/images/"
    upload_bucket_csv = f"{upload_bucket_base}/csv/"

    # Parse the arguments
    parse = ArgumentParser()
    parse.add_argument("--progress_file", type=str, default="all_mover_progress.txt")
    args = parse.parse_args()
    tracking_file = args.progress_file

    # Mover ids start at 1
    current_index = load_progress(tracking_file, default=1)
    already_downloaded = []
    if os.path.exists("already_downloaded.txt"):
        with open("already_downloaded.txt", "r") as f:
            already_downloaded = f.read().split("\n")[:-1]
    for i in tqdm(range(current_index, TOTAL_MOVERS)):
        success, already_downloaded = download_mover_data(
            str(i),
            BASE,
            bad_request_output,
            centered_image_folder,
            whole_image_folder,
            META_DATA_COLUM_NAMES,
            POSITION_TABLE_COLUMN_NAMES,
            MOVER_TABLE_COLUMN_NAMES,
            image_csv_path,
            mover_csv_path,
            position_csv_path,
            already_downloaded,
        )
        if success:
            save_progress(tracking_file, i)
            np.savetxt("already_downloaded.txt", already_downloaded, fmt="%s")
            time.sleep(1)
        else:
            print(f"Print Bad Request at {i} !!!!")
            print("Stopping")

            # Uploading
            os.system(f"gsutil -m cp -n -r {IMAGE_FOLDER} {upload_bucket_images}")
            time.sleep(2)
            os.system(f"gsutil -m cp -r {CSV_FOLDER} {upload_bucket_csv}")
            time.sleep(2)

            break
