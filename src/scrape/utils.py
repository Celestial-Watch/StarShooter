import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List
import time
from config import BASE, SESSION_ID


def save_progress(file: str, index: int):
    """
    Save the index of the last mover downloaded
    """
    print(f"Saving progress: {index}")
    with open(file, "w") as f:
        f.write(str(index))


def load_progress(file: str, default: int = 0):
    """
    Get the index of the last mover downloaded
    """
    if os.path.exists(file):
        with open(file, "r") as f:
            return int(f.read())
    else:
        return default


def extract_mover_id_tag(soup: BeautifulSoup) -> tuple:
    """
    Extract mover_id and tag
    """
    mover_id, tag = (
        str(soup.findAll("h3")[-1])
        .replace("<h3>", "")
        .replace("</h3>", "")
        .replace("Mover ", "")
        .split(" - ")
    )
    return mover_id, tag


def get_centered_on_asteroid_image_links(soup: BeautifulSoup) -> List[str]:
    """
    Get the links to the images that are centered on the asteroid
    """
    # Get the index of the image row
    contents = soup.findAll("tr")
    image_row_idx = -1
    for idx, content in enumerate(contents):
        if content.find("td").find("br"):
            image_row_idx = idx + 1
            break

    image_links = [img["src"] for img in contents[image_row_idx].findAll("img")[1:]]
    return image_links


def download_images(
    image_links: List[str], base: str, output_dir: str, sleep: int = 1
) -> None:
    """
    Download the images

    Args:
        image_links: List of image links
        base: Base url
        output_dir: Directory to save the images
        sleep: Time to sleep between downloads
    """
    for image_link in image_links:
        image_name = image_link.split("/")[-1]
        os.system(f"wget {base + image_link} -O {output_dir}/{image_name}")
        time.sleep(sleep)


def get_image_meta_data(
    soup: BeautifulSoup,
    file_names: List[str],
    meta_data_columns: List[str],
    position_columns: List[str],
    mover_id: str,
) -> pd.DataFrame:
    tables = soup.findAll("tr")
    meta_data = [
        [data.text for data in table.find_all("td")[1:]]
        for table in tables
        if ".fit" in table.text
    ]

    position_data = [
        [data.text for data in table.find_all("td")[1:]]
        for table in tables
        if "position" in str(table)
    ]

    # Combine all the data
    file_names_df = pd.DataFrame(file_names, columns=["file_name"])
    meta_data_df = pd.DataFrame(meta_data, columns=meta_data_columns)
    position_df = pd.DataFrame(position_data, columns=position_columns)
    image_meta_data_df = pd.concat([file_names_df, meta_data_df, position_df], axis=1)
    image_meta_data_df["mover_id"] = mover_id

    # Make mover_id the first column
    cols = list(image_meta_data_df.columns)
    cols = [cols[-1]] + cols[:-1]
    image_meta_data_df = image_meta_data_df[cols]

    return image_meta_data_df


def get_mover_data(
    index: str,
    base: str,
    bad_request_output: str,
    output_dir: str,
    meta_data_columns: List[str],
    position_columns: List[str],
    image_csv_path: str,
    mover_csv_path: str,
    sleep: int = 1,
) -> bool:
    """
    Returns whether the request was successful or not
    """
    # Get html
    url = base + index
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")

    # Whether request was successful
    if str(soup.prettify()) == bad_request_output:
        return False

    # Get the data
    mover_id, tag = extract_mover_id_tag(soup)
    image_links = get_centered_on_asteroid_image_links(soup)
    download_images(image_links, base, output_dir, sleep)
    file_names = [link.split("/")[-1] for link in image_links]
    image_meta_data_df = get_image_meta_data(
        soup, file_names, meta_data_columns, position_columns, mover_id
    )
    mover_label_df = pd.DataFrame(
        [[mover_id, tag, index]], columns=["mover_id", "label_tag", "totas_id"]
    )

    # Save the data
    with open(image_csv_path, "a") as f:
        image_meta_data_df.to_csv(f, header=False, index=False)

    with open(mover_csv_path, "a") as f:
        mover_label_df.to_csv(f, header=False, index=False)

    return True


def download_whole_image(
    url: str, id: str, output_dir: str, base: str = BASE, session_id: str = SESSION_ID
):
    r = requests.get(url, headers={"Cookie": f"PHPSESSID={session_id}"})
    soup = BeautifulSoup(r.content, "html.parser")
    img = soup.findAll("img")[-1]
    img_url = base + img["src"]
    os.system(f"wget {img_url} -O {output_dir}/{id}.png")


def get_n_centered_on_asteroid(
    url: str,
    mover_id: str,
    mover_image_csv: str,
    output_dir: str,
    base: str = BASE,
    cookie_token: str = SESSION_ID,
    required_images: int = 4,
    sleep: int = 1,
):
    r = requests.get(url, headers={"Cookie": cookie_token})
    soup = BeautifulSoup(r.content, "html.parser")

    # Get the data
    image_links = get_centered_on_asteroid_image_links(soup)
    if len(image_links) == required_images:
        download_images(image_links, base, output_dir, sleep)
        
        # Write to csv
        image_names = [link.split("/")[-1] for link in image_links]
        with open(mover_image_csv, "a") as f:
            for image in image_names:
                f.write(f"{mover_id},{image}\n")
    else:
        print(f"Skipping {mover_id}")
    