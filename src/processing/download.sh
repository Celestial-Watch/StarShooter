# Set the GCP bucket and folder path
BUCKET_NAME="mlp-asteroid-data"
FOLDER_PATH="alistair/images/"
DOWNLOAD_FOLDER="data/alistair/images"

# Create the download folder
mkdir -p ${DOWNLOAD_FOLDER}

# Download all files in the folder
gsutil -m cp -r gs://${BUCKET_NAME}/${FOLDER_PATH}* ${DOWNLOAD_FOLDER}

echo "Download completed!"