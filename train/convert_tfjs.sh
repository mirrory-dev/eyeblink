#!/bin/bash

ROOT=$(dirname "$0")

tensorflowjs_converter \
   --input_format keras \
   --output_format tfjs_graph_model \
   --quantization_bytes 2 \
   ${ROOT}/models/latest.h5 \
   ${ROOT}/models/tfjs
cp ./models/tfjs/* ../models/