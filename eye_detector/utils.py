import os

import cv2
import dlib
import numpy as np
import onnxruntime as ort
from imutils import face_utils

IMG_SIZE = (34, 26)

base_path = os.path.dirname(os.path.abspath(__file__))
predictor = dlib.shape_predictor(os.path.join(base_path, "models", "shape_predictor_68_face_landmarks.dat"))

# for improving the performance I have used ONNX Runtime
model_onnx = ort.InferenceSession(os.path.join(base_path, "models", "model.onnx"),
                                  providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])


def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int32)

    eye_img = img[eye_rect[1]: eye_rect[3], eye_rect[0]: eye_rect[2]]

    return eye_img, eye_rect


def blink_detector(img, bbox):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get the landmarks/parts for the face
    shapes = predictor(gray, dlib.rectangle(*bbox))
    shapes = face_utils.shape_to_np(shapes)

    # get the left eye points
    eye_img_l, _ = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, _ = crop_eye(gray, eye_points=shapes[42:48])

    # resize the eye images
    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    # concatenate the eye images
    eye_input_l = (eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.0)
    eye_input_r = (eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.0)

    # predict the eye state
    pred_l = model_onnx.run(None, {model_onnx.get_inputs()[0].name: eye_input_l})
    pred_r = model_onnx.run(None, {model_onnx.get_inputs()[0].name: eye_input_r})

    return pred_l[0], pred_r[0]
