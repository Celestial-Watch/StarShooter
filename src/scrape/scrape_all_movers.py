import time
from tqdm import tqdm
import os
from argparse import ArgumentParser
from utils import save_progress, load_progress, get_mover_data
from config import BASE, META_DATA_COLUM_NAMES, POSITION_TABLE_COLUMN_NAMES

TOTAL_MOVERS = 400_000
MOVER_BASE = f"{BASE}/mover.php?id="
image_table_labels = META_DATA_COLUM_NAMES[1:]
position_table_labels = POSITION_TABLE_COLUMN_NAMES[1:]
bad_request_output = "<p>\n nice try.\n <br/>\n logged.\n <br/>\n bye.\n</p>\n"

if __name__ == "__main__":
    data_folder = os.path.abspath("./../../data")
    csv_folder = f"{data_folder}/csv"

    image_csv = "all_movers_images.csv"
    image_csv_path = f"{csv_folder}/{image_csv}"
    if not os.path.exists(image_csv_path):
        with open(image_csv_path, "w") as f:
            f.write(
                "mover_id,file_name,"
                + ",".join(image_table_labels)
                + ",".join(position_table_labels)
                + "\n"
            )

    mover_csv = "all_movers.csv"
    mover_csv_path = f"{csv_folder}/{mover_csv}"
    if not os.path.exists(mover_csv_path):
        with open(mover_csv_path, "w") as f:
            f.write("mover_id,label_tag,totas_id\n")

    relative_image_folder = "images/all_movers"
    image_folder = f"{data_folder}/{relative_image_folder}"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # GS definitions
    upload_bucket_base = "gs://mlp-asteroid-data"
    upload_bucket_images = f"{upload_bucket_base}/{relative_image_folder}/"
    upload_bucket_csv = f"{upload_bucket_base}/csv/"

    # Parse the arguments
    parse = ArgumentParser()
    parse.add_argument("--progress_file", type=str, default="all_mover_progress.txt")
    args = parse.parse_args()
    tracking_file = args.progress_file

    current_index = load_progress(tracking_file, default=1)
    for i in tqdm(range(current_index, TOTAL_MOVERS)):
        if get_mover_data(
            str(i),
            MOVER_BASE,
            bad_request_output,
            image_folder,
            image_table_labels,
            position_table_labels,
            image_csv_path,
            mover_csv_path,
        ):
            save_progress(tracking_file, i)
            time.sleep(1)
        else:
            print(f"Print Bad Request at {i} !!!!")
            print("Stopping")

            # Uploading
            os.system(f"gsutil -m cp -n -r {image_folder} {upload_bucket_images}")
            time.sleep(2)
            os.system(f"gsutil -m cp -r {csv_folder} {upload_bucket_csv}")
            time.sleep(2)

            break
