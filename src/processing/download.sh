# Set the GCP bucket and folder path
BUCKET_NAME="mlp-asteroid-data"
FOLDER_PATH="alistair/images/"
DOWNLOAD_FOLDER="data/alistair/images"

# Create the download folder
mkdir -p ${DOWNLOAD_FOLDER}

# Download only the files listed in data/images_not_found.csv
while IFS= read -r file; do
    gsutil -m cp gs://${BUCKET_NAME}/${FOLDER_PATH}/${file}.png ${DOWNLOAD_FOLDER}
done < data/images_not_found.csv

echo "Download completed!"