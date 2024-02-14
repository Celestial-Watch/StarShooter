# Download data
if [ ! -d "data" ]; then
    mkdir data
fi

if [ "$1" = "all" ]; then
  echo "Downloading all data..."
  gsutil -m cp -r "gs://mlp-asteroid-data/*" data
else if [ "$1" = "new" ]; then
  echo "Downloading all new data..."
  gsutil -m cp -r -n "gs://mlp-asteroid-data/30x30_images" data
else
  echo "Downloading only the 30x30 images and lookup files..."
  gsutil -m cp -r -n "gs://mlp-asteroid-data/30x30_images" data 
  gsutil -m cp -r \
    "gs://mlp-asteroid-data/mover_images_lookup.csv" \
    "gs://mlp-asteroid-data/rejected_mover_images_lookup.csv" \
    data
fi

