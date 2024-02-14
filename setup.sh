#!/bin/bash
# Setup python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download data
mkdir data
cd data
gsutil -m cp -r \
  "gs://mlp-asteroid-data/30x30_images" \
  "gs://mlp-asteroid-data/mover_images_lookup.csv" \
  "gs://mlp-asteroid-data/rejected_mover_images_lookup.csv" \
  .


