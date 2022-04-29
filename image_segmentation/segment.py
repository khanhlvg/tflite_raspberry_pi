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
"""Main script to run image segmentation."""

import argparse
import sys
import time
from typing import List

import cv2
import numpy as np

import utils
from image_segmenter import ImageSegmenter, ColoredLabel, OutputType
from image_segmenter import ImageSegmenterOptions


def run(model: str, display_mode: str, num_threads: int, enable_edgetpu: bool,
        camera_id: int, width: int, height: int) -> None:
  """Continuously run inference on images acquired from the camera.

    Args:
        model: Name of the TFLite image segmentation model.
        display_mode: Name of mode to display image segmentation.
        num_threads: Number of CPU threads to run the model.
        enable_edgetpu: Whether to run the model on EdgeTPU.
        camera_id: The camera id to be passed to OpenCV.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.
    """

  # Initialize the image segmentation model.
  options = ImageSegmenterOptions(
    num_threads=num_threads,
    enable_edgetpu=enable_edgetpu)
  segmenter = ImageSegmenter(model_path=model, options=options)

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  fps_avg_frame_count = 10

  # Continuously capture images from the camera and run inference.
  while cap.isOpened():
    start_time1 = time.time()
    success, image = cap.read()
    if not success:
      sys.exit(
        'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
    end_time1 = time.time() - start_time1
    print(f'Capture time: {end_time1 * 1000}ms')

    counter += 1
    image = cv2.flip(image, 1)

    start_time1 = time.time()
    # Segment with each frame from camera.
    segmentation_result = segmenter.segment(image)
    end_time1 = time.time() - start_time1
    print(f'Inference time: {end_time1 * 1000}ms')

    start_time1 = time.time()
    # Convert the segmentation result into an image.
    seg_map_img, found_colored_labels = utils.segmentation_map_to_image(segmentation_result)

    # Resize the segmentation mask to be the same shape as input image.
    seg_map_img = cv2.resize(seg_map_img,
                             dsize=(image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    end_time1 = time.time() - start_time1
    print(f'Conversion time: {end_time1 * 1000}ms')

    # Visualize segmentation result on image.
    start_time1 = time.time()
    overlay = visualize(image, seg_map_img, display_mode, fps, found_colored_labels)
    end_time1 = time.time() - start_time1
    print(f'Visualization time: {end_time1 * 1000}ms')
    print()

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('image_segmentation', overlay)

  cap.release()
  cv2.destroyAllWindows()


def visualize(input_image: np.ndarray,
              segmentation_map_image: np.ndarray,
              display_mode: str,
              fps: float,
              colored_labels: List[ColoredLabel]) -> np.ndarray:
  """Visualize segmentation result on image.

    Args:
        input_image: The [height, width, 3] RGB input image.
        segmentation_map_image: The [height, width, 3] RGB segmentation map image.
        display_mode: How the segmentation map should be shown. 'overlay' or 'side-by-side'.
        fps: Value of fps.
        colored_labels: List of colored labels found in the segmentation result.

    Returns:
        Overlay image with segmentation result.
    """
  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  padding_top, padding_bottom = 0, 0
  padding_left, padding_right = 0, 150
  w_rect, h_rect = 20, 20
  label_margin = 10
  alpha = 0.5

  # Show the input image and the segmentation map image.
  if display_mode == 'overlay':
    # Overlay mode.
    overlay = cv2.addWeighted(input_image, alpha, segmentation_map_image, alpha, 0)
  elif display_mode == 'side-by-side':
    # Side by side mode.
    overlay = cv2.hconcat([input_image, segmentation_map_image])
  else:
    sys.exit(
      f'ERROR: Unsupported display mode: {display_mode}.'
    )

  # Show the FPS
  fps_text = 'FPS = ' + str(int(fps))
  text_location = (left_margin, row_size)
  cv2.putText(overlay, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
              font_size, text_color, font_thickness)

  # Initialize the origin coordinates of the label.
  w_show = overlay.shape[1] + label_margin
  h_show = overlay.shape[0] // row_size + label_margin

  # Expand the frame to show the label.
  overlay = cv2.copyMakeBorder(overlay, padding_top, padding_bottom,
                               padding_left, padding_right, cv2.BORDER_CONSTANT)

  # Show the label on top-right frame.
  for colored_label in colored_labels:
    rect_color = colored_label.color
    start_point = (w_show, h_show)
    end_point = (w_show + w_rect, h_show + h_rect)
    cv2.rectangle(overlay, start_point, end_point,
                  rect_color, -font_thickness)

    label_location = (w_show + w_rect + label_margin, h_show + label_margin)
    cv2.putText(overlay, colored_label.label, label_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
    h_show += (h_rect + label_margin)

  return overlay


def main():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    '--model',
    help='Name of image segmentation model.',
    required=False,
    default='deeplabv3.tflite')
  parser.add_argument(
    '--displayMode',
    help='Mode to display image segmentation.',
    required=False,
    default='overlay')
  parser.add_argument(
    '--numThreads',
    help='Number of CPU threads to run the model.',
    required=False,
    default=4)
  parser.add_argument(
    '--enableEdgeTPU',
    help='Whether to run the model on EdgeTPU.',
    action="store_true",
    required=False,
    default=False)
  parser.add_argument(
    '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
    '--frameWidth',
    help='Width of frame to capture from camera.',
    required=False,
    default=640)
  parser.add_argument(
    '--frameHeight',
    help='Height of frame to capture from camera.',
    required=False,
    default=480)
  args = parser.parse_args()

  run(args.model, args.displayMode, int(args.numThreads), bool(args.enableEdgeTPU),
      int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
  main()
