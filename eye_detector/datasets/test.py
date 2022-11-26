import asyncio
import logging
import os

import cv2
import dlib
import numpy as np
import tensorflow as tf
from imutils import face_utils
from imutils.paths import list_images

from mtcnnort import MTCNN

IMG_SIZE = (34, 26)

mtcnn = MTCNN()
predictor = dlib.shape_predictor('models/model.dat')
model = tf.keras.models.load_model('models/model_batch_256.h5')
blink_threshold = 0.5


async def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int32)

    eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


async def blink_detector(img, bbox):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get the landmarks/parts for the face
    shapes = predictor(img, dlib.rectangle(*bbox))
    shapes = face_utils.shape_to_np(shapes)

    # get the left eye points
    eye_img_l, _ = await crop_eye(img, eye_points=shapes[36:42])
    eye_img_r, _ = await crop_eye(img, eye_points=shapes[42:48])

    # resize the eye images
    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    # concatenate the eye images
    eye_input_l = (eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.0)
    eye_input_r = (eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.0)

    pred_l = model.predict(eye_input_l)
    pred_r = model.predict(eye_input_r)

    return pred_l, pred_r


async def detector(image):
    try:
        logging.info(f"\nProcessing image: {image}")
        img = cv2.imread(image)
        bbox, _ = mtcnn.detect_faces_raw(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        x1, y1, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])
        left_eye, right_eye = await blink_detector(img, [x1, y1, x2, y2])
        logging.info(f"Left eye: {left_eye}, Right eye: {right_eye}")

        folder_name = "/".join(image.split("/")[:-1])
        if left_eye < blink_threshold or right_eye < blink_threshold:
            logging.info("Blink detected")
            if not os.path.exists(f'blink/{folder_name}'):
                os.makedirs(f'blink/{folder_name}')
            cv2.imwrite(f'blink/{folder_name}/{image.split("/")[-1]}', img)
            return

        if not os.path.exists(f'no_blink/{folder_name}'):
            os.makedirs(f'no_blink/{folder_name}')
        cv2.imwrite(f'no_blink/{folder_name}/{image.split("/")[-1]}', img)

    except Exception as ex:
        print(f"Error processing image: {image}, {ex}")


async def main():
    images = list_images("~/Downloads/3kk_only_successful")
    tasks = [asyncio.create_task(detector(image)) for image in images]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
