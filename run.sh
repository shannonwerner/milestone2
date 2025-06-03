#!/bin/bash

set -e

cd -- "$( dirname -- "${BASH_SOURCE[0]}")"
source .venv/bin/activate
pip3 install -r requirements.txt
python3 data_preprocessing.py
python3 gan.py