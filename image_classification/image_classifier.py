# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cv2
import os
import zipfile
from typing import NamedTuple, List

try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf
  Interpreter = tf.lite.Interpreter

class Category(NamedTuple):
    """A result of a image classification."""
    label: str
    prob: float

class ImageClassifier(object):
    """A wrapper class for a TFLite image classification model."""

    def __init__(self, model_name: str) -> None:
        """Initialize a image classification model.

        Args:
            model_name: Path of the TFLite image classification model.
        """
        
        # Append TFLITE extension to model_name if there's no extension
        _, ext = os.path.splitext(model_name)
        if not ext:
            model_name += '.tflite'

        interpreter = Interpreter(model_path=model_name, num_threads=4)
        interpreter.allocate_tensors()

        self._input_index = interpreter.get_input_details()[0]['index']
        self._output_index = interpreter.get_output_details()[0]['index']

        self._input_height = interpreter.get_input_details()[0]['shape'][1]
        self._input_width = interpreter.get_input_details()[0]['shape'][2]

        self._is_quantized_model = interpreter.get_input_details()[0]['dtype'] == np.uint8
        
        # Load label list from metadata.
        try:
            with zipfile.ZipFile(model_name) as model_with_metadata:
                if not model_with_metadata.namelist():
                    raise ValueError('Invalid TFLite model: no label file found.')

                file_name = model_with_metadata.namelist()[0]
                with model_with_metadata.open(file_name) as label_file:
                    label_list = label_file.read().splitlines()
                    self._labels_list = [label.decode('ascii') for label in label_list]
        except zipfile.BadZipFile:
            print(
                'ERROR: Please use models trained with Model Maker or downloaded from TensorFlow Hub.'
            )

        self._interpreter = interpreter


    def _set_input_tensor(self, image: np.ndarray) -> None:
        """Sets the input tensor."""
        tensor_index = self._input_index
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the input image as required by the TFLite model."""
        image = cv2.resize(image, (self._input_width, self._input_height))
        return image


    def classify_image(self, image: np.ndarray) -> List[Category]:
        """Run classification on an input.
        Args:
            image: A [height, width, 3] RGB image.
        Returns: A list of prediction result. Sorted by probability descending.
        """
        image = self._preprocess(image)
        self._set_input_tensor(image)
        self._interpreter.invoke()
        output_details = self._interpreter.get_output_details()[0]
        output = np.squeeze(self._interpreter.get_tensor(self._output_index))

        # If the model is quantized (uint8 data), then dequantize the results
        if self._is_quantized_model:
            scale, zero_point = output_details['quantization']
            output = scale * (output - zero_point)

        # Sort output by probability descending.
        prob_descending = sorted(
            range(len(output)), key=lambda k: output[k], reverse=True)

        return [Category(label=self._labels_list[idx], prob=output[idx]) for idx in prob_descending]