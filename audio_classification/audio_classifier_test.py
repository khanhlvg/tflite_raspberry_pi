# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests for the AudioClassifier wrapper."""

import csv
import unittest
from typing import List

from tflite_support.task import audio
from tflite_support.task import core
from tflite_support.task import processor
from audio_classifier import Category
import numpy as np

_MODEL_FILE = 'yamnet.tflite'
_GROUND_TRUTH_FILE = 'test_data/ground_truth.csv'
_AUDIO_FILE = 'test_data/meow_16k.wav'
_ACCEPTABLE_ERROR_RANGE = 0.01


class AudioClassifierTest(unittest.TestCase):

  def setUp(self):
    """Initialize the shared variables."""
    super().setUp()
    self._load_ground_truth()
    

    # Load the TFLite model to get the audio format required by the model.
    self._initialize_model()

    # Load the input audio file. Use only the beginning of the file that fits
    # the model input size.
    audio_tensor = audio.TensorAudio.create_from_wav_file(_AUDIO_FILE, self._classifier.required_input_buffer_size)

    self._input_tensor = audio_tensor
        
  def _initialize_model(self,
                        max_results: int = 5,
                        score_threshold: float = 0.0,
                        label_allow_list: List[str] = None,
                        label_deny_list: List[str] = None) -> None:
    base_options = core.BaseOptions(file_name=_MODEL_FILE)
    classification_options = processor.ClassificationOptions(max_results=max_results,
                                                             score_threshold=score_threshold,
                                                             class_name_allowlist=label_allow_list,
                                                             class_name_denylist=label_deny_list)
    options = audio.AudioClassifierOptions(base_options=base_options, classification_options=classification_options)

    # AudioClassifier
    self._classifier = audio.AudioClassifier.create_from_options(options)
    
  def _parse(self, classification_results) -> List[Category]:
    """Parse the output classification_results into  a list of Category instances."""
    categories = [Category(label=category.class_name,score=category.score)
                  for category in classification_results.classifications[0].classes]
    
    return categories
  
  def test_default_option(self):
    """Check if the default option works correctly."""
    classification_results = self._classifier.classify(self._input_tensor)
    categories = self._parse(classification_results)

    # Check if all ground truth classification is found.
    for gt_classification in self._ground_truth_classifications:
      is_gt_found = False
      for real_classification in categories:
        is_label_match = real_classification.label == gt_classification.label
        is_score_match = abs(real_classification.score -
                             gt_classification.score) < _ACCEPTABLE_ERROR_RANGE

        # If a matching classification is found, stop the loop.
        if is_label_match and is_score_match:
          is_gt_found = True
          break

      # If no matching classification found, fail the test.
      self.assertTrue(is_gt_found, '{0} not found.'.format(gt_classification))

  def test_allow_list(self):
    """Test the label_allow_list option."""
    allow_list = ['Cat']
    self._initialize_model(label_allow_list=allow_list)
    classification_results = self._classifier.classify(self._input_tensor)
    categories = self._parse(classification_results)

    for category in categories:
      label = category.label
      self.assertIn(
          label, allow_list,
          'Label "{0}" found but not in label allow list'.format(label))

  def test_deny_list(self):
    """Test the label_deny_list option."""
    deny_list = ['Animal']
    self._initialize_model(label_deny_list=deny_list)
    classification_results = self._classifier.classify(self._input_tensor)
    categories = self._parse(classification_results)

    for category in categories:
      label = category.label
      self.assertNotIn(label, deny_list,
                       'Label "{0}" found but in deny list.'.format(label))

  def test_score_threshold_option(self):
    """Test the score_threshold option."""
    score_threshold = 0.5
    self._initialize_model(score_threshold=score_threshold)
    classification_results = self._classifier.classify(self._input_tensor)
    categories = self._parse(classification_results)

    for category in categories:
      score = category.score
      self.assertGreaterEqual(
          score, score_threshold,
          'Classification with score lower than threshold found. {0}'.format(
              category))

  def test_max_results_option(self):
    """Test the max_results option."""
    max_results = 3
    self._initialize_model(max_results=max_results)
    classification_results = self._classifier.classify(self._input_tensor)
    categories = self._parse(classification_results)

    self.assertLessEqual(
        len(categories), max_results, 'Too many results returned.')

  def _load_ground_truth(self):
    """Load ground truth classification result from a CSV file."""
    self._ground_truth_classifications = []
    with open(_GROUND_TRUTH_FILE) as f:
      reader = csv.DictReader(f)
      for row in reader:
        category = Category(label=row['label'], score=float(row['score']))

        self._ground_truth_classifications.append(category)

# pylint: disable=g-unreachable-test-method

  def _create_ground_truth_csv(self, output_file=_GROUND_TRUTH_FILE):
    """A util function to regenerate the ground truth result.

    This function is not used in the test but it exists to make adding more
    audio and ground truth data to the test easier in the future.

    Args:
      output_file: Filename to write the ground truth CSV.
    """
    classification_results = self._classifier.classify(self._input_tensor)
    categories = self._parse(classification_results)
    with open(output_file, 'w') as f:
      header = ['label', 'score']
      writer = csv.DictWriter(f, fieldnames=header)
      writer.writeheader()
      for category in categories:
        writer.writerow({
            'label': category.label,
            'score': category.score,
        })


# pylint: enable=g-unreachable-test-method

if __name__ == '__main__':
  unittest.main()
