import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

base = "https://totas.cosmos.esa.int/mover.php?id="

mover_links = pd.read_csv("test_data_set.csv")


# Let's write a function that does all of the above
def get_photo_meta_data(name: str, link: str, base: str):
    r = requests.get(base + link)
    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.find_all("tr")
    images_tables = [i for i in tables if ".fit" in i.text]
    images_paths = [
        (i.find_all("a")[0]["href"], *[j.text for j in i.find_all("td")])
        for i in images_tables
    ]
    labels = [
        "Link",
        "FileName",
        "ExposureBegin",
        "ExposureTime",
        "CcdTemperature",
        "BackgroundMean",
        "BackgroundSigma",
        "RightAscension",
        "Declination",
        "Angle",
        "Azimuth",
        "Altitude",
        "Airmass",
        "MagnitudeZeroPoint",
        "PixelScaleX",
        "PixelScaleY",
        "NumberOfDetections",
        "NumberOfStars",
        "AverageResidual",
        "RmsResidual",
        "FitOrder",
        "OffsetRightAscension",
        "OffsetDeclination",
        "Offset",
    ]
    images_df = pd.DataFrame(images_paths, columns=labels)
    positions_tables = [
        list(map(lambda x: x.text, i.find_all("td")[1:]))
        for i in tables
        if "position" in str(i)
    ]
    positions_labels = ["X", "Y", "Flux", "RightAscension", "Declination", "Magnitude"]
    positions_df = pd.DataFrame(positions_tables, columns=positions_labels)
    merged_df = pd.concat([images_df, positions_df], axis=1)
    merged_df["Name"] = name
    return merged_df


data = list(zip(mover_links["mover_id"].unique(), mover_links["totas_id"].unique()))
meta_data = pd.concat(
    [
        get_photo_meta_data(name, str(link), base)
        for name, link in tqdm(data, total=len(data))
    ]
)

meta_data.to_csv("test_movers_meta_data.csv", index=False)
