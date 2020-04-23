#!/bin/bash

cd train
poetry install
python3 train.py
./convert_tfjs.sh
cp ./models/tfjs/* ../models/
