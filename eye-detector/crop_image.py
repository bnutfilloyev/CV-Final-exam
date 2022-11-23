import os
import csv

import cv2
import dlib
import numpy as np
import pandas as pd
from imutils import face_utils
from imutils.paths import list_images

from mtcnnort.mtcnn_ort import MTCNN
from sklearn.model_selection import train_test_split

detector = MTCNN()
predictor = dlib.shape_predictor('models/model.dat')
IMG_SIZE = (34, 26)


def read_csv(path):
    width = 34
    height = 26
    dims = 1

    with open(path, 'r') as f:
        # read the scv file with the dictionary format
        reader = csv.DictReader(f)
        rows = list(reader)

    # imgs is a numpy array with all the images
    # tgs is a numpy array with the tags of the images
    imgs = np.empty((len(list(rows)), height, width, dims), dtype=np.uint8)
    tgs = np.empty((len(list(rows)), 1))

    for row, i in zip(rows, range(len(rows))):
        # convert the list back to the image format
        img = row['image']
        img = img.strip('[').strip(']').split(', ')
        im = np.array(img, dtype=np.uint8)
        im = im.reshape((height, width))
        im = np.expand_dims(im, axis=2)
        imgs[i] = im

        # the tag for open is 1 and for close is 0
        tag = row['state']
        if tag == 'open':
            tgs[i] = 1
        else:
            tgs[i] = 0

    # shuffle the dataset
    index = np.random.permutation(imgs.shape[0])
    imgs = imgs[index]
    tgs = tgs[index]

    # return images and their respective tags
    return imgs, tgs


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

    eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


def crop_image():
    for image in list_images('close-eye (412)'):
        print("Image -> ", image)
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bbox, _ = detector.detect_faces_raw(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if bbox.shape[0] == 0:
            print("No face detected")
            continue
        x1, y1, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])
        landmarks = predictor(gray, dlib.rectangle(x1, y1, x2, y2))
        shapes = face_utils.shape_to_np(landmarks)

        # get the left and right eye images
        eye_img_l, _ = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, _ = crop_eye(gray, eye_points=shapes[42:48])

        eye_img_r = cv2.flip(eye_img_r, flipCode=1)
        try:
            cv2.imwrite(os.path.join('images', 'closed_eyes', 'left' + '-' + os.path.basename(image)), eye_img_l)
            cv2.imwrite(os.path.join('images', 'closed_eyes', 'right' + '-' + os.path.basename(image)), eye_img_r)
        except Exception as e:
            print(e)
            print(bbox)


def image_to_csv():
    df = pd.read_csv('dataset.csv')

    for image in list_images('images/closed_eyes'):
        img = cv2.imread(image)
        img = cv2.resize(img, dsize=IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_list = []

        for row in img:
            for pixel in row:
                image_list.append(pixel)

        image_arr = str(image_list)

        label = 'open' if 'open_eyes' in image else 'close'

        # df = pd.concat([df, pd.DataFrame({'image': [image_arr], 'state': [label]})])
        df = df.append({'state': label, 'image': image_arr}, ignore_index=True)
        # break

    df.to_csv('dataset.csv', index=False)


def split_dataset():
    X, y = read_csv('dataset.csv')

    n_total = len(X)
    X_result = np.empty((n_total, 26, 34, 1))

    for i, x in enumerate(X):
        print("Image -> ", i)
        img = x.reshape((26, 34, 1))

        X_result[i] = img

    x_train, x_val, y_train, y_val = train_test_split(X_result, y, test_size=0.1, random_state=42)

    np.save('models/x_train.npy', x_train)
    np.save('models/y_train.npy', y_train)
    np.save('models/x_val.npy', x_val)
    np.save('models/y_val.npy', y_val)


if __name__ == '__main__':
    # crop_image()
    # image_to_csv()
    split_dataset()
