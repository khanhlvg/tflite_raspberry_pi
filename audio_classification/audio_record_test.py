"""Tests for audio_record."""

import unittest
from unittest import mock

import numpy as np
from numpy import testing

import audio_record

_CHANNELS = 1
_SAMPLING_RATE = 16000
_BUFFER_SIZE = 15600


class AudioRecordTest(unittest.TestCase):

  def setUp(self):
    super().setUp()

    # Mock sounddevice.InputStream
    with mock.patch("sounddevice.InputStream") as mock_input_stream_new_method:
      self.mock_input_stream = mock.MagicMock()
      mock_input_stream_new_method.return_value = self.mock_input_stream
      self.record = audio_record.AudioRecord(_CHANNELS, _SAMPLING_RATE,
                                             _BUFFER_SIZE)

      # Save the initialization arguments of InputStream for later assertion.
      _, self.init_args = mock_input_stream_new_method.call_args

  def test_init_args(self):
    # Assert parameters of InputStream initialization
    self.assertEqual(
      self.init_args["channels"], _CHANNELS,
      "InputStream's channels doesn't match the initialization argument.")
    self.assertEqual(
      self.init_args["samplerate"], _SAMPLING_RATE,
      "InputStream's samplerate doesn't match the initialization argument.")

  def test_life_cycle(self):
    # Assert start recording routine.
    self.record.start_recording()
    self.mock_input_stream.start.assert_called_once()

    # Assert stop recording routine.
    self.record.stop()
    self.mock_input_stream.stop.assert_called_once()

  def test_buffer_data(self):
    callback_fn = self.init_args["callback"]

    # Create dummy data to feed to the AudioRecord instance.
    chunk_size = int(_BUFFER_SIZE * 0.5)
    input_data = []
    for _ in range(3):
      dummy_data = np.random.rand(chunk_size, 1).astype(float)
      input_data.append(dummy_data)
      callback_fn(dummy_data)

    # Assert read data of a single chunk.
    recorded_audio_data = self.record.read(chunk_size)
    testing.assert_almost_equal(recorded_audio_data, input_data[-1])

    # Assert read all data in buffer.
    recorded_audio_data = self.record.read(chunk_size * 2)
    print(input_data[-2].shape)
    expected_data = np.concatenate(input_data[-2:])
    testing.assert_almost_equal(recorded_audio_data, expected_data)

    # Assert exception if read too much data.
    with self.assertRaises(ValueError):
      self.record.read(chunk_size * 3)


if __name__ == "__main__":
  unittest.main()
