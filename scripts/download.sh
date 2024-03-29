# Download data
mkdir -p data

if [ "$1" = "all" ]; then
  echo "Downloading all data..."
  gsutil -m cp -r "gs://mlp-asteroid-data/*" data
elif [ "$1" = "new" ]; then
  echo "Downloading new centered on asteroid images..."
  gsutil -m cp -r -n "gs://mlp-asteroid-data/images/centered_on_asteroid" data
else
  echo "Downloading only the centered on asteroid images and lookup files..."
  gsutil -m cp -r -n "gs://mlp-asteroid-data/images/centered_on_asteroid" data 
  gsutil -m cp -r "gs://mlp-asteroid-data/csv" data
fi
