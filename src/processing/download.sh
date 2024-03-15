# Set the GCP bucket and folder path
BUCKET_NAME="mlp-asteroid-data"
FOLDER_PATH="final/csv/"
DOWNLOAD_FOLDER="data/alistair/csv/"

# Create the download folder
mkdir -p ${DOWNLOAD_FOLDER}

# Create a temporary text file with the list of files to delete
# cat data/images_not_found.csv | sed 's/$/.png/' > delete_list.txt

# Delete all the files listed in delete_list.txt from DOWNLOAD_FOLDER
# cat delete_list.txt | xargs -I {} rm -f ${DOWNLOAD_FOLDER}/{}

# Remove the temporary text file
#rm -f delete_list.txt

# Download the files from the GCP bucket
gsutil -m cp -rn gs://${BUCKET_NAME}/${FOLDER_PATH}* ${DOWNLOAD_FOLDER}

echo "Download completed!"