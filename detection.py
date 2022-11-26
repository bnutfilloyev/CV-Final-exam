import base64
import logging

import cv2
import numpy as np

from eye_detector.utils import blink_detector
from mtcnnort.mtcnn_ort import MTCNN
from occlusion import prep_model, prep_image, detect, check_face_validity
from schemas import *
from utils.torch_utils import select_device


def fix_b64_padding(b64_string):
    return f"{b64_string}{'=' * (len(b64_string) % 4)}"


def base64_2_bytestr(image_str: str):
    if "base64," in image_str:
        image_str = image_str.split("base64,")[1]
    image_str = base64.urlsafe_b64decode(fix_b64_padding(image_str))
    return image_str


class Detector:
    def __init__(self):
        self.mtcnn = MTCNN()
        self.device = select_device()
        self.weights = 'weights/real_120.pt'
        self.data = 'RS_REAL.yaml'
        self.model = prep_model(self.weights, self.device, self.data)

    def eye_detect(self, byte_image, blink_threshold=0.5):
        cv_image = cv2.imdecode(np.frombuffer(byte_image, np.uint8), cv2.IMREAD_COLOR)
        bbox, _ = self.mtcnn.detect_faces_raw(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        x1, y1, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])
        left_eye, right_eye = blink_detector(cv_image, [x1, y1, x2, y2])
        logging.info(f"Left eye: {left_eye}, Right eye: {right_eye}")
        return EyeDetectResponse(
            is_blinking=bool(left_eye < blink_threshold and right_eye < blink_threshold),
            results=EyeResult(left_eye=left_eye, right_eye=right_eye)
        )

    def occlusion_detect(self, byte_image):
        cv_image = cv2.imdecode(np.frombuffer(byte_image, np.uint8), cv2.IMREAD_COLOR)
        image_processed = prep_image(cv_image, self.device)
        predictions = detect(image_processed, self.model)
        validity, results = check_face_validity(predictions, image_processed, cv_image)
        logging.info(f"Face validity: {validity}, Results: {results}, Predictions: {predictions}")
        return OcclusionDetectResponse(is_occluded=not validity)


def read_image():
    with open('image.txt', 'r') as f:
        return f.read()


detector = Detector()

if __name__ == "__main__":
    image_str = read_image()
    image = base64_2_bytestr(image_str)
    eye = detector.eye_detect(image)
    occlusion = detector.occlusion_detect(image)
    print(f"Eye: {eye}, Occlusion: {occlusion}")
