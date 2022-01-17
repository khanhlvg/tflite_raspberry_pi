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
"""A module to record audio in a streaming basis."""
import threading
import numpy as np
import sounddevice as sd


class AudioRecord(object):

  def __init__(self, channels, sampling_rate: int) -> None:
    self._lock = threading.Lock()
    self._audio_buffer = []

    def audio_callback(indata, frames, time, status):
      self._lock.acquire()
      self._audio_buffer.append(np.copy(indata))
      self._lock.release()

    # Create an input stream to continuously capture the audio data.
    self._stream = sd.InputStream(
      channels=channels,
      samplerate=sampling_rate,
      callback=audio_callback,
    )

  def start_recording(self) -> None:
    """Start the input stream"""
    self._stream.start()

  def stop(self) -> None:
    """Stop the input stream"""
    self._audio_buffer = []
    self._stream.stop()

  @property
  def buffer(self) -> np.ndarray:
    self._lock.acquire()
    if len(self._audio_buffer):
      result = np.concatenate(self._audio_buffer)
      self._audio_buffer.clear()
    else:
      result = np.zeros((0, 1))
    self._lock.release()

    return result
