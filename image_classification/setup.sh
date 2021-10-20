#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./"
else
  DATA_DIR="$1"
fi

# Install Python dependencies.
python3 -m pip install -r requirements.txt

# Download MobileNet TF Lite model and labels.
FILE=${DATA_DIR}/mobilenet_v1_1.0_224_quant_and_labels.zip
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip' \
    -o ${FILE}
fi
# Unzip the TF Lite model and labels
unzip mobilenet_v1_1.0_224_quant_and_labels.zip -d ${DATA_DIR}

# Remove zip file
rm mobilenet_v1_1.0_224_quant_and_labels.zip

# Download MobileNet TF Lite edgetpu model.
FILE=${DATA_DIR}/mobilenet_v1_1.0_224_quant_edgetpu.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://dl.google.com/coral/canned_models/mobilenet_v1_1.0_224_quant_edgetpu.tflite' \
    -o ${FILE}
fi

echo -e "Downloaded files are in ${DATA_DIR}"