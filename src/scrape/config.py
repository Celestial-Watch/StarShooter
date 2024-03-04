import os

# Requests
BASE = "https://totas.cosmos.esa.int/"
MOVER_BASE = BASE + "mover.php?id="
SESSION_ID = "4968f2db96e72ca5b760ebe20418b651"

# Structure
MOVER_TABLE_COLUMN_NAMES = ["mover_id", "sohas_id", "label"]
POSITION_TABLE_COLUMN_NAMES = [
    "mover_id",
    "image_id",
    "centered_image_id",
    "X",
    "Y",
    "Flux",
    "RightAscension",
    "Declination",
    "Magnitude",
]
META_DATA_COLUM_NAMES = [
    "image_id",
    "link",
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

# File paths
DATA_FOLDER = os.path.abspath("./../../data")
CSV_FOLDER = f"{DATA_FOLDER}/csv"
IMAGE_FOLDER = f"{DATA_FOLDER}/images"
