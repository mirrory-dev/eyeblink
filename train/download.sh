#!/bin/bash

poetry install

curl -sL https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2 | bzip2 -d > shape_predictor_68_face_landmarks.dat