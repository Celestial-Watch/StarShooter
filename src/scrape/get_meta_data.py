import requests
import pandas as pd
from bs4 import BeautifulSoup
from config import BASE, POSITION_TABLE_COLUMN_NAMES, META_DATA_COLUM_NAMES, CSV_FOLDER
from utils import extract_image_meta_data
from typing import List, Optional


def get_photo_meta_data(
    mover_id: str,
    totas_id: str,
    base: str,
    file_names: List[str],
    meta_data_columns: Optional[List[str]] = META_DATA_COLUM_NAMES,
    position_columns: Optional[List[str]] = POSITION_TABLE_COLUMN_NAMES,
):
    r = requests.get(base + f"/mover.php?id={totas_id}")
    soup = BeautifulSoup(r.text, "html.parser")

    extract_image_meta_data(
        soup, file_names, meta_data_columns, position_columns, mover_id
    )

    tables = soup.find_all("tr")
    images_tables = [i for i in tables if ".fit" in i.text]
    images_paths = [
        (i.find_all("a")[0]["href"], *[j.text for j in i.find_all("td")])
        for i in images_tables
    ]
    labels = ["Link"] + META_DATA_COLUM_NAMES
    images_df = pd.DataFrame(images_paths, columns=labels)

    positions_tables = [
        list(map(lambda x: x.text, i.find_all("td")[1:]))
        for i in tables
        if "position" in str(i)
    ]
    positions_labels = POSITION_TABLE_COLUMN_NAMES[1:]
    positions_df = pd.DataFrame(positions_tables, columns=positions_labels)
    merged_df = pd.concat([images_df, positions_df], axis=1)
    merged_df["Name"] = mover_id
    return merged_df


csv_file = CSV_FOLDER + "/movers_cond_12.csv"
mover_links = pd.read_csv(csv_file).head(3)
meta_data = pd.concat(
    [
        get_photo_meta_data(name, link, BASE)
        for name, link in zip(mover_links["mover_id"], mover_links["totas_id"])
    ]
)
print(meta_data)
