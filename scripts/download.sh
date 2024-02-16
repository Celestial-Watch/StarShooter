# Download data
mkdir -p data

if [ "$1" = "all" ]; then
  echo "Downloading all data..."
  gsutil -m cp -r "gs://mlp-asteroid-data/*" data
elif [ "$1" = "new" ]; then
  echo "Downloading all new data..."
  gsutil -m cp -r -n "gs://mlp-asteroid-data/30x30_images" data
else
  echo "Downloading only the 30x30 images and lookup files..."
  gsutil -m cp -r -n "gs://mlp-asteroid-data/30x30_images" data 
  gsutil -m cp -r "gs://mlp-asteroid-data/csv/*" data
fi
