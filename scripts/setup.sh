#!/bin/bash
# Setup python
if [ -d ".venv" ]; then
    echo "Virtual environment already setup. Skipping python setup."
else
    echo "Setting up python virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi

# Download data
./download.sh $1


