import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List, Tuple
import time
from config import BASE, SESSION_ID
import subprocess
import numpy as np


def save_progress(file: str, index: int) -> None:
    """
    Save the index of the last mover downloaded

    Args:
        file (str): The file to save the index to
        index (int): The index to save
    """
    with open(file, "w") as f:
        f.write(str(index))


def load_progress(file: str, default: int = 0) -> int:
    """
    Get the index of the last mover downloaded

    Args:
        file (str): The file to get the index from
        default (int, optional): The default index. Defaults to 0.

    Returns: The index of the last mover downloaded
    """
    if os.path.exists(file):
        with open(file, "r") as f:
            return int(f.read())
    else:
        return default


def extract_sohas_id_tag(soup: BeautifulSoup) -> Tuple[str, str]:
    """
    Extract sohas_id and tag

    Args:
        soup (BeautifulSoup): The soup of the page

    Returns: Tuple of sohas_id and tag
    """
    cleaned_str = (
        str(soup.findAll("h3")[-1])
        .replace("<h3>", "")
        .replace("</h3>", "")
        .replace("Mover ", "")
        .replace("\n", "")
    )
    if " - " not in cleaned_str:
        return cleaned_str, ""
    sohas_id, tag = cleaned_str.split(" - ", maxsplit=1)
    return sohas_id, tag


def get_centered_on_asteroid_image_links(soup: BeautifulSoup) -> List[str]:
    """
    Get the links to the images that are centered on the asteroid

    Args:
        soup (BeautifulSoup): The soup of the page

    Returns: List of image links
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
        image_links (List[str]): The links to the images
        base (str): The base url
        output_dir (str): The directory to save the images
        sleep (int, optional): Time to sleep between downloads. Defaults to 1.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for image_link in image_links:
        image_name = image_link.split("/")[-1]
        subprocess.run(
            f"wget {base + image_link} -O {output_dir}/{image_name}",
            shell=True,
            capture_output=True,
        )
        time.sleep(sleep)


def extract_image_data(
    soup: BeautifulSoup,
    meta_data_columns: List[str],
) -> pd.DataFrame:
    """
    Extract the meta data of the images froum the soup

    Args:
        soup: The soup of the page
        meta_data_columns: The columns of the meta data

    Returns: DataFrame of the image meta data
    """
    tables = soup.findAll("tr")
    meta_data = [
        [
            table.find_all("a")[0]["href"].split("=")[-1],  # image_id
            table.find_all("a")[0]["href"],  # link
            *[data.text for data in table.find_all("td")[1:]],
        ]
        for table in tables
        if ".fit" in table.text
    ]

    # Combine all the data
    image_meta_data_df = pd.DataFrame(meta_data, columns=meta_data_columns)

    return image_meta_data_df


def extract_position_meta_data(
    soup: BeautifulSoup,
    position_data_columns: List[str],
    mover_id: str,
    centered_image_names: str,
    whole_image_names: str,
) -> pd.DataFrame:
    """
    Extract the position data of the images froum the soup

    Args:
        soup: The soup of the page
        position_data_columns: The columns of the position data
        mover_id: The mover id
        centered_image_names: The names of the centered on asteroid images
        whole_image_names: The names of the whole images
    """
    tables = soup.findAll("tr")

    position_data = [
        [data.text for data in table.find_all("td")[1:]]
        for table in tables
        if "position" in str(table)
    ]

    # Combine all the data
    position_df = pd.DataFrame(position_data, columns=position_data_columns[3:])
    mover_ids = [mover_id] * len(centered_image_names)

    image_info = np.array([mover_ids, whole_image_names, centered_image_names]).T
    image_info_df = pd.DataFrame(
        image_info,
        columns=position_data_columns[:3],
    )
    position_df = pd.concat([image_info_df, position_df], axis=1)

    return position_df


def extract_image_meta_data(
    soup: BeautifulSoup,
    file_names: List[str],
    meta_data_columns: List[str],
    position_columns: List[str],
    mover_id: str,
) -> pd.DataFrame:
    """
    Extract the meta data of the images froum the soup

    Args:
        soup: The soup of the page
        file_names: The names of the centered on asteroid images
        meta_data_columns: The columns of the meta data
        position_columns: The columns of the position data
        mover_id: The mover id

    Returns: DataFrame of the image meta data
    """
    tables = soup.findAll("tr")
    meta_data = [
        [
            table.find_all("a")[0]["href"].split("=")[-1],  # image_id
            *[data.text for data in table.find_all("td")[1:]],
        ]
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


def download_mover_data(
    mover_id: str,
    base: str,
    bad_request_output: str,
    centered_image_folder: str,
    whole_image_folder: str,
    meta_data_columns: List[str],
    position_columns: List[str],
    mover_columns: List[str],
    image_csv_path: str,
    mover_csv_path: str,
    position_csv_path: str,
    already_downloaded: List[str],
    sleep: int = 1,
) -> Tuple[bool, List[str]]:
    """
    Download images of a mover centered on the asteroid and save the meta data of the images

    Args:
        mover_id (str): The mover id, e.g. 123
        base (str): The base url
        bad_request_output (str): The output of the bad request
        centered_image_folder (str): The directory to save the centered on asteroid images
        whole_image_folder (str): The directory to save the whole images
        meta_data_columns (List[str]): The columns of the meta data
        position_columns (List[str]): The columns of the position data
        mover_columns (List[str]): The columns of the mover data
        image_csv_path (str): The csv file to write the image meta data to
        mover_csv_path (str): The csv file to write the mover data to
        position_csv_path (str): The csv file to write the position data to
        already_downloaded (List[str]): List of Ids of whole images that have already been downloaded
        sleep (int, optional): Time to sleep between downloads. Defaults to 1.

    Returns whether the request was successful or not and the list of already downloaded images
    """
    # Get html
    url = base + f"/mover.php?id={mover_id}"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")

    # Whether request was successful
    if str(soup.prettify()) == bad_request_output:
        return False

    # Get the data
    sohas_id, tag = extract_sohas_id_tag(soup)
    image_links = get_centered_on_asteroid_image_links(soup)
    download_images(image_links, base, centered_image_folder, sleep)
    centered_image_ids = [link.split("/")[-1] for link in image_links]

    image_df = extract_image_data(soup, meta_data_columns)
    whole_image_ids = image_df["image_id"].tolist()
    position_df = extract_position_meta_data(
        soup, position_columns, mover_id, centered_image_ids, whole_image_ids
    )
    mover_label_df = pd.DataFrame([[mover_id, sohas_id, tag]], columns=mover_columns)

    # Save the data
    with open(image_csv_path, "a") as f:
        # ignore already downloaded images
        image_df = image_df[~image_df["image_id"].isin(already_downloaded)]
        for idx in range(len(image_df)):
            image_id = image_df["image_id"].iloc[idx]
            already_downloaded.append(image_id)
            link = download_whole_image(
                f"{base}/image.php?id={image_id}", image_id, whole_image_folder, base
            )
            image_df.loc[idx, "link"] = link

        image_df.to_csv(f, header=False, index=False)

    with open(mover_csv_path, "a") as f:
        mover_label_df.to_csv(f, header=False, index=False)

    with open(position_csv_path, "a") as f:
        position_df.to_csv(f, header=False, index=False)

    return (True, already_downloaded)


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
    Download images of a mover centered on the asteroid and save the meta data of the images

    Args:
        index (str): The index of the mover
        base (str): The base url
        bad_request_output (str): The output of the bad request
        output_dir (str): The directory to save the images
        meta_data_columns (List[str]): The columns of the meta data
        position_columns (List[str]): The columns of the position data
        image_csv_path (str): The csv file to write the image meta data to
        mover_csv_path (str): The csv file to write the mover data to
        sleep (int, optional): Time to sleep between downloads. Defaults to 1.

    Returns whether the request was successful or not
    """
    # Get html
    url = base + f"/mover.php?id={index}"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")

    # Whether request was successful
    if str(soup.prettify()) == bad_request_output:
        return False

    # Get the data
    mover_id, tag = extract_sohas_id_tag(soup)
    image_links = get_centered_on_asteroid_image_links(soup)
    download_images(image_links, base, output_dir, sleep)
    file_names = [link.split("/")[-1] for link in image_links]
    image_meta_data_df = extract_image_meta_data(
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
) -> str:
    """
    Gets full resolution images from the url

    Args:
        url (str): The url to scrape
        id (str): The id of the mover
        output_dir (str): The directory to save the images
        base (str, optional): The base url. Defaults to BASE.
        session_id (str, optional): The session id. Defaults to SESSION_ID.

    Returns: The url of the image
    """
    r = requests.get(url, headers={"Cookie": f"PHPSESSID={session_id}"})
    soup = BeautifulSoup(r.content, "html.parser")

    img = soup.findAll("img")[-1]
    if not img["src"].startswith("image-view.php"):
        print(f"No image available for image id {id}")
        return ""
    img_url = base + img["src"]
    subprocess.run(
        f"wget {img_url} -O {output_dir}/{id}.png",
        shell=True,
        capture_output=True,
    )
    return img_url


def get_n_centered_on_asteroid(
    url: str,
    mover_id: str,
    mover_image_csv: str,
    output_dir: str,
    base: str = BASE,
    required_images: int = 4,
    sleep: int = 1,
) -> None:
    """
    Get images centered on the asteroid from url if there are required_images number of images

    Args:
        url (str): The url to scrape
        mover_id (str): The mover id
        mover_image_csv (str): The csv file to write the image names to
        output_dir (str): The directory to save the images
        base (str, optional): The base url. Defaults to BASE.
        required_images (int, optional): The number of images required. Defaults to 4.
        sleep (int, optional): Time to sleep between downloads. Defaults to 1.
    """
    r = requests.get(url)
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


def get_mover_list(condition: int, csv_file: str, base: str = BASE):
    """
    Gets a mapping from mover_id to totas_id

    Args:
        condition (int): The condition to filter the movers
        csv_file (str): The file to write the data to
        base (str, optional): The base url. Defaults to BASE.
    """
    url = f"{base}/index.php?list=Movers&cond={condition}"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")

    mover_table = soup.findAll("table")[-1]
    mover_rows = mover_table.findAll("tr")[1:]

    with open(csv_file, "w") as f:
        f.write("mover_id,totas_id\n")
        for mover_row in mover_rows:
            totas_id = mover_row.find("a")["href"].split("=")[-1]
            mover_id = mover_row.find("a").text
            f.write(f"{mover_id},{totas_id}\n")
