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

from image_segmenter import ColoredLabel, Segmentation, OutputType
from typing import List


def segmentation_map_to_image(segmentation: Segmentation) -> (np.ndarray, List[ColoredLabel]):
  """Convert the segmentation result into a RGB image.

    Params:
      segmentation: An output of a image segmentation model.

    Returns:
      seg_map_img: The visualized segmentation result as an RGB image.
      found_colored_labels: The list of ColoredLabels found in the image.
  """
  if segmentation.output_type != OutputType.CATEGORY_MASK:
    raise ValueError('Unsupported output type. Only OutputType.CATEGORY_MASK is supported.')

  # Get the list of unique labels from the model output.
  found_label_indices, inverse_map, counts = np.unique(segmentation.masks, return_inverse=True, return_counts=True)
  count_dict = dict(zip(found_label_indices, counts))

  # Sort the list of unique label so that the class with the most pixel will come first.
  sorted_label_indices = sorted(found_label_indices, key=lambda index: count_dict[index], reverse=True)
  found_colored_labels = [
    segmentation.colored_labels[idx] for idx in sorted_label_indices
  ]

  # Convert segmentation map into RGB image of the same size as the input image.
  # Note: We use the inverse map to avoid running the heavy loop in Python and pass
  # it over to Numpy's C++ implementation to improve performance.
  found_colors = [
    segmentation.colored_labels[idx].color for idx in found_label_indices
  ]
  output_shape = [segmentation.masks.shape[0], segmentation.masks.shape[1], 3]
  seg_map_img = np.array(found_colors)[inverse_map].reshape(output_shape).astype(np.uint8)

  return seg_map_img, found_colored_labels
