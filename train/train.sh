#!/bin/bash

poetry install
python3 preprocess.py
python3 train.py
./convert_tfjs.sh
