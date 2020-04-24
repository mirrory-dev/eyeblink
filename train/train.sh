#!/bin/bash

poetry install
python3 preprocess.py
mkdir -p models
python3 train.py
./convert_tfjs.sh
