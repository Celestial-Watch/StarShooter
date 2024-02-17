import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import os
from argparse import ArgumentParser
from typing import List

image_csv_path = "csv/image.csv"
# images contain -> mover_id, file_name, resolution, meta_data
mover_csv_path = "csv/mover.csv"
# mover_id, label_tag, totas_id

TOTAL_MOVERS = 400_000


base = "https://totas.cosmos.esa.int/mover.php?id="
cookie_token = ""

image_table_labels = [
    "exposure_begin",
    "exposure_time",
    "ccd_temperature",
    "background_mean",
    "background_sigma",
    "right_ascension",
    "declination",
    "angle",
    "azimuth",
    "altitude",
    "airmass",
    "magnitude_zero_point",
    "pixel_scale_x",
    "pixel_scale_y",
    "number_of_detections",
    "number_of_stars",
    "average_residual",
    "rms_residual",
    "fit_order",
    "offset_right_ascension",
    "offset_declination",
    "offset",
]

position_table_labels = [
    "x",
    "y",
    "flux",
    "right_ascension_",
    "declination_",
    "magnitude",
]

bad_request_output = "<p>\n nice try.\n <br/>\n logged.\n <br/>\n bye.\n</p>\n"


def get_image(
    base: str,
    index: str,
    bad_request_output: str,
    image_table_labels: List[str],
    position_table_labels: List[str],
    image_csv_path: str,
    mover_csv_path: str,
) -> bool:
    url = base + index
    print(url)
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")

    if str(soup.prettify()) == bad_request_output:
        return False

    print(soup.findAll("h3"))

    mover_id, tag = (
        str(soup.findAll("h3")[-1])
        .replace("<h3>", "")
        .replace("</h3>", "")
        .replace("Mover ", "")
        .split(" - ")
    )
    contents = soup.findAll("tr")
    for i, content in enumerate(contents):
        if content.find("td").find("br"):
            break

    # Get the image list
    image_list = [img["src"] for img in contents[i + 1].findAll("img")[1:]]
    image_names = [link.split("/")[-1] for link in image_list]

    # Download the images
    for i, image in enumerate(image_list):
        os.system(f"wget {base + image} -O all_mover_images/{image_names[i]}")
        time.sleep(1)

    file_names_df = pd.DataFrame(image_names, columns=["file_name"])

    tables = soup.findAll("tr")
    image_tables = [i for i in tables if ".fit" in i.text]
    image_meta_data = [[j.text for j in i.find_all("td")][1:] for i in image_tables]

    image_meta_data_df = pd.DataFrame(image_meta_data, columns=image_table_labels)

    positions_tables = [
        list(map(lambda x: x.text, i.find_all("td")[1:]))
        for i in tables
        if "position" in str(i)
    ]

    positions_df = pd.DataFrame(positions_tables, columns=position_table_labels)

    image_meta_data_df = pd.concat(
        [file_names_df, image_meta_data_df, positions_df], axis=1
    )
    image_meta_data_df["mover_id"] = mover_id

    # Make mover_id the first column
    cols = list(image_meta_data_df.columns)
    cols = [cols[-1]] + cols[:-1]
    image_meta_data_df = image_meta_data_df[cols]

    mover_label_df = pd.DataFrame(
        [[mover_id, tag, url]], columns=["mover_id", "label_tag", "totas_id"]
    )

    # Save the data
    with open(image_csv_path, "a") as f:
        image_meta_data_df.to_csv(f, header=False, index=False)

    with open(mover_csv_path, "a") as f:
        mover_label_df.to_csv(f, header=False, index=False)

    return True


def save_progress(index: int, tracking_file: str):
    with open(tracking_file, "w") as f:
        f.write(str(index))


def load_progress(tracking_file: str):
    if os.path.exists(tracking_file):
        with open(tracking_file, "r") as f:
            return int(f.read())
    else:
        return 1


if __name__ == "__main__":
    parse = ArgumentParser()
    parse.add_argument("--progress_file", type=str, default="all_mover_progress.txt")

    args = parse.parse_args()
    tracking_file = args.progress_file

    upload_bucket_images = "gs://mlp-asteroid-data/all_mover_images/"
    upload_bucket_csv = "gs://mlp-asteroid-data/csv/"
    upload_checker = 0

    current_index = load_progress(tracking_file)
    for i in tqdm(range(current_index, TOTAL_MOVERS)):
        if get_image(
            base,
            str(i),
            bad_request_output,
            image_table_labels,
            position_table_labels,
            image_csv_path,
            mover_csv_path,
        ):
            save_progress(i, tracking_file)
            time.sleep(1)
            upload_checker += 1

        else:
            print(f"Print Bad Request at {i} !!!!")
            print("Stopping")

            # Uploading
            os.system(f"gsutil -m cp -n -r all_mover/images/ {upload_bucket_images}")
            time.sleep(2)
            os.system(f"gsutil -m cp -n -r csv/ {upload_bucket_csv}")
            time.sleep(2)

            break
