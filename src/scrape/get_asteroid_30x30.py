from tqdm import tqdm
import pandas as pd
import time
import os
from argparse import ArgumentParser
import subprocess
from utils import (
    save_progress,
    load_progress,
    get_mover_data,
)
from config import (
    BASE,
    IMAGE_FOLDER,
    CSV_FOLDER,
    META_DATA_COLUM_NAMES,
    POSITION_TABLE_COLUMN_NAMES,
)

image_table_labels = META_DATA_COLUM_NAMES[1:]
position_table_labels = ["pos_" + label for label in POSITION_TABLE_COLUMN_NAMES[1:]]
bad_request_output = "<p>\n nice try.\n <br/>\n logged.\n <br/>\n bye.\n</p>\n"

if __name__ == "__main__":
    parse = ArgumentParser()
    parse.add_argument("--mover_file", type=str, default="movers.csv")
    parse.add_argument("--progress_file", type=str, default="scraper_progress.txt")

    args = parse.parse_args()

    mover_totas_csv = pd.read_csv(f"{CSV_FOLDER}/{args.mover_file}")
    tracking_file = args.progress_file

    upload_bucket = "gs://mlp-asteroid-data/csv/"
    upload_checker = 0

    mover_image_csv_file = args.mover_file.split("/")[-1]
    meta_data_csv = f"{mover_image_csv_file.split('.')[0]}_image_meta_data.csv"
    meta_data_csv_path = f"{CSV_FOLDER}/{meta_data_csv}"
    if not os.path.exists(meta_data_csv_path):
        with open(meta_data_csv_path, "w") as f:
            f.write(
                "mover_id,file_name,"
                + ",".join(image_table_labels)
                + ","
                + ",".join(position_table_labels)
                + "\n"
            )

    mover_csv = f"{mover_image_csv_file.split('.')[0]}_mover.csv"
    mover_csv_path = f"{CSV_FOLDER}/{mover_csv}"
    if not os.path.exists(mover_csv_path):
        with open(mover_csv_path, "w") as f:
            f.write("mover_id,label_tag,totas_id\n")

    relative_image_folder = "centered_on_asteroid"
    image_folder = f"{IMAGE_FOLDER}/{relative_image_folder}"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    current_index = load_progress(tracking_file)
    for idx in tqdm(range(current_index, len(mover_totas_csv))):
        mover_id, totas_id = mover_totas_csv.iloc[idx]
        get_mover_data(
            str(totas_id),
            BASE,
            bad_request_output,
            image_folder,
            image_table_labels,
            position_table_labels,
            meta_data_csv_path,
            mover_csv_path,
        )
        save_progress(tracking_file, idx)
        time.sleep(1)
        upload_checker += 1

        if upload_checker == 10:
            subprocess.run(f"gsutil -m cp -r {CSV_FOLDER} {upload_bucket}", shell=True)
            upload_checker = 0
            time.sleep(2)
