# Set the GCP bucket and folder path
BUCKET_NAME="mlp-asteroid-data"
FOLDER_PATH="data/images/centered_on_asteroid/"
DOWNLOAD_FOLDER="data/alistair/30x30_images"

# Create the download folder
mkdir -p ${DOWNLOAD_FOLDER}

# Download all files in the folder
gsutil -m cp -rn gs://${BUCKET_NAME}/${FOLDER_PATH}* ${DOWNLOAD_FOLDER}

echo "Download completed!"