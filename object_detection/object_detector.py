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
"""Code to run a TFLite object detection model."""

import cv2
import numpy as np
from typing import List, NamedTuple
import zipfile

# pylint: disable=g-import-not-at-top
try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
  from tflite_runtime.interpreter import load_delegate
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf

  Interpreter = tf.lite.Interpreter
  load_delegate = tf.lite.experimental.load_delegate
# pylint: enable=g-import-not-at-top

import utils


class ObjectDetectorOptions(NamedTuple):
  """A config to initialize an object detector."""

  enable_edgetpu: bool = False
  """Enable the model to run on EdgeTPU."""

  label_allow_list: List[str] = None
  """The optional allow list of labels."""

  label_deny_list: List[str] = None
  """The optional deny list of labels."""

  max_results: int = -1
  """The maximum number of top-scored detection results to return."""

  num_threads: int = 1
  """The number of threads to be used for TFLite ops that support
  multi-threading when running inference with CPU."""

  score_threshold: float = 0.0
  """The score threshold of detection results to return."""


class Rect(NamedTuple):
  """A rectangle in 2D space."""
  left: float
  top: float
  right: float
  bottom: float


class Category(NamedTuple):
  """A result of a classification task."""
  label: str
  score: float
  index: int


class Detection(NamedTuple):
  """A detected object as the result of an ObjectDetector."""
  bounding_box: Rect
  categories: List[Category]


class ObjectDetector:
  """A wrapper class for a TFLite object detection model."""

  _mean = 127.5
  """Default mean normalization parameter for float model."""
  _std = 127.5
  """Default std normalization parameter for float model."""

  def __init__(self, model_path: str,
               options: ObjectDetectorOptions = ObjectDetectorOptions()) -> None:
    """Initialize a TFLite object detection model.
    Args:
        model_path: Path to the TFLite model.
        options: The config to initialize an object detector. (Optional)
    """

    # Load label list from metadata.
    try:
      with zipfile.ZipFile(model_path) as model_with_metadata:
        if len(model_with_metadata.namelist()) == 0:
          raise ValueError('Invalid TFLite model: no label file found.')

        file_name = model_with_metadata.namelist()[0]
        with model_with_metadata.open(file_name) as label_file:
          label_list = label_file.read().splitlines()
          self._label_list = [label.decode('ascii') for label in label_list]
    except zipfile.BadZipFile:
      print('ERROR: Please use models trained with Model Maker or downloaded from TensorFlow Hub.')
      raise ValueError('Invalid TFLite model: no metadata found.')

    # Initialize TFLite model.
    if options.enable_edgetpu:
      if utils.edgetpu_lib_name() is None:
        raise Exception("The current OS isn't supported by Coral EdgeTPU.")
      interpreter = Interpreter(model_path=model_path,
                                experimental_delegates=[load_delegate(utils.edgetpu_lib_name())],
                                num_threads=options.num_threads)
    else:
      interpreter = Interpreter(model_path=model_path, num_threads=options.num_threads)

    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    self._input_size = input_detail['shape'][2], input_detail['shape'][1]
    self._is_quantized_input = input_detail['dtype'] == np.uint8
    self._interpreter = interpreter
    self._options = options

  def detect(self, input_image: np.ndarray) -> List[Detection]:
    """Run detection on an input image.
    Args:
        input_image: A [height, width, 3] RGB image. Note that height and width
          can be anything since the image will be immediately resized according
          to the needs of the model within this function.
    Returns:
        A Person instance.
    """
    image_height, image_width, _ = input_image.shape

    input_tensor = self._preprocess(input_image)

    self._set_input_tensor(input_tensor)
    self._interpreter.invoke()

    # Get all output details
    boxes = self._get_output_tensor(0)
    classes = self._get_output_tensor(1)
    scores = self._get_output_tensor(2)
    count = int(self._get_output_tensor(3))

    return self._postprocess(
      boxes,
      classes,
      scores,
      count,
      image_width,
      image_height
    )

  def _preprocess(self, input_image: np.ndarray) -> np.ndarray:
    """Preprocess the input image as required by the TFLite model."""

    # Resize the input
    input_tensor = cv2.resize(input_image, self._input_size)

    # Normalize the input if it's a float model (aka. not quantized)
    if self._is_quantized_input is False:
      input_tensor = (np.float32(input_tensor) - self._mean) / self._std

    # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor

  def _set_input_tensor(self, image):
    """Sets the input tensor."""
    tensor_index = self._interpreter.get_input_details()[0]['index']
    input_tensor = self._interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

  def _get_output_tensor(self, index):
    """Returns the output tensor at the given index."""
    output_details = self._interpreter.get_output_details()[index]
    tensor = np.squeeze(self._interpreter.get_tensor(output_details['index']))
    return tensor

  def _postprocess(self, boxes: np.ndarray,
                   classes: np.ndarray,
                   scores: np.ndarray,
                   count: int,
                   image_width: int,
                   image_height: int
                   ) -> List[Detection]:
    results = []

    # Parse the model output into a list of Detection entities.
    for i in range(count):
      if scores[i] >= self._options.score_threshold:
        y_min, x_min, y_max, x_max = boxes[i]
        bounding_box = Rect(
          top=int(y_min * image_height),
          left=int(x_min * image_width),
          bottom=int(y_max * image_height),
          right=int(x_max * image_width)
        )
        class_id = int(classes[i])
        category = Category(
          score=scores[i],
          label=self._label_list[class_id],  # 0 is reserved for background
          index=class_id
        )
        result = Detection(
          bounding_box=bounding_box,
          categories=[category]
        )
        results.append(result)

    # Sort detection results by score ascending
    sorted_results = sorted(
      results, key=lambda detection: detection.categories[0].score, reverse=True
    )

    # Filter out detections in deny list
    filtered_results = sorted_results
    if self._options.label_deny_list is not None:
      filtered_results = list(filter(
        lambda detection: detection.categories[0].label not in self._options.label_deny_list,
        filtered_results
      ))

    # Keep only detections in allow list
    if self._options.label_allow_list is not None:
      filtered_results = list(filter(
        lambda detection: detection.categories[0].label in self._options.label_allow_list,
        filtered_results
      ))

    # Only return maximum of max_results detection.
    if self._options.max_results > 0:
      result_count = min(len(filtered_results), self._options.max_results)
      filtered_results = filtered_results[:result_count]

    return filtered_results
