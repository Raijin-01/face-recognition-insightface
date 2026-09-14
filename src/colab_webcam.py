from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from IPython.display import Javascript, display
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow

from .recognizer import FaceRecognizer


JS_CODE = r'''
async function captureFrame() {
  const video = document.createElement('video');
  const stream = await navigator.mediaDevices.getUserMedia({video: true});
  document.body.appendChild(video);
  video.srcObject = stream;
  await video.play();
  await new Promise(resolve => setTimeout(resolve, 500));
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  stream.getTracks().forEach(track => track.stop());
  video.remove();
  return canvas.toDataURL('image/jpeg', 0.9);
}
captureFrame();
'''


def capture_frame() -> np.ndarray:
    """Capture one webcam frame from a Google Colab browser session."""
    data_url = eval_js(JS_CODE)
    image_bytes = base64.b64decode(data_url.split(",", 1)[1])
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not decode the webcam frame.")
    return image


def recognize_webcam_frame(recognizer: FaceRecognizer) -> np.ndarray:
    image = capture_frame()
    for (x1, y1, x2, y2), name, score in recognizer.recognize_image(image):
        label = f"{name} ({score:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2_imshow(image)
    return image
