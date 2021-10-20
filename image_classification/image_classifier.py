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

    def __init__(self, model_path: str, label_file: str) -> None:
        """Initialize a image classification model.

        Args:
            model_name: Name of the TFLite image classification model.
            label_file: Path of the label list file.
        """

        interpreter = Interpreter(model_path=model_path, num_threads=4)
        interpreter.allocate_tensors()

        self._input_index = interpreter.get_input_details()[0]['index']
        self._output_index = interpreter.get_output_details()[0]['index']

        self._input_height = interpreter.get_input_details()[0]['shape'][1]
        self._input_width = interpreter.get_input_details()[0]['shape'][2]

        self._is_quantized_model = interpreter.get_input_details()[0]['dtype'] == np.uint8

        self._interpreter = interpreter

        self._labels_list = self._load_labels(label_file)

    def _load_labels(self, label_path: str) -> List[str]:
        """Load label list from file.
        Args:
            label_path: Full path of label file.
        Returns: An array contains the list of labels.
        """
        with open(label_path, 'r') as f:
            return [line.strip() for _, line in enumerate(f.readlines())]


    def _set_input_tensor(self, image: np.ndarray) -> None:
        """Sets the input tensor."""
        tensor_index = self._input_index
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the input image as required by the TFLite model."""
        image = cv2.resize(image, (self._input_width, self._input_height))
        return image


    def classify_image(self, image: np.ndarray, top_k: int = 3) -> List[Category]:
        """Run classification on an input.
        Args:
            image: A [height, width, 3] RGB image.
            top_k: max classification results.
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

        return [Category(label=self._labels_list[idx], prob=output[idx]) for idx in prob_descending[:top_k]]