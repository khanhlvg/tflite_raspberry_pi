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
"""Unit test of object detection using ObjectDetector wrapper."""

import csv
import unittest
from unittest import mock

import cv2
import detect

_MODEL_FILE = 'efficientdet_lite0.tflite'
_GROUND_TRUTH_FILE = 'test_data/table_results.csv'
_IMAGE_FILE = 'test_data/table.jpg'
_BBOX_IOU_THRESHOLD = 0.9
_ALLOW_LIST = ['knife', 'cup']
_DENY_LIST = ['book']
_SCORE_THRESHOLD = 0.3
_MAX_RESULTS = 3


class ObjectDetectorTest(unittest.TestCase):

  def setUp(self):
    """Initialize the shared variables."""
    super().setUp()
    self.image = cv2.imread(_IMAGE_FILE)
    print(self.image.shape, self.image.dtype)

  @mock.patch("detect.cv2.VideoCapture", return_value=mock.MagicMock())
  @mock.patch("detect.cv2.imshow", return_value=mock.MagicMock())
  @mock.patch("detect.cv2.waitKey", return_value=27)
  def test_object_detection(self, _, mock_imshow, mock_video_capture):
    """Check if the default option works correctly."""

    # Mock cv2.VideoCapture to return preset image.
    mock_video_capture.isOpened = mock.MagicMock(return_value=True)
    mock_video_capture.read = mock.MagicMock(return_value=(True, self.image))

    # Mock cv2.imshow to get the result image.
    img = None
    def fake_imshow(_, output_image):
      print(output_image.shape)
    mock_imshow.return_value = fake_imshow

    detect.run(
      model=_MODEL_FILE,
      camera_id=0,
      width=0,
      height=0,
      num_threads=4,
      enable_edgetpu=False
    )

# pylint: disable=g-unreachable-test-method

  def _create_groud_truth_csv(self, output_file=_GROUND_TRUTH_FILE):
    """A util function to recreate the ground truth result."""
    detector = od.ObjectDetector(_MODEL_FILE)
    result = detector.detect(self.image)
    with open(output_file, 'w') as f:
      header = ['label', 'left', 'top', 'right', 'bottom', 'score']
      writer = csv.DictWriter(f, fieldnames=header)
      writer.writeheader()
      for d in result:
        writer.writerow({
            'label': d.categories[0].label,
            'left': d.bounding_box.left,
            'top': d.bounding_box.top,
            'right': d.bounding_box.right,
            'bottom': d.bounding_box.bottom,
            'score': d.categories[0].score,
        })


# pylint: enable=g-unreachable-test-method

if __name__ == '__main__':
  unittest.main()