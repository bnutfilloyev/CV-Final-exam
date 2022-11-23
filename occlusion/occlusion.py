import cv2
import numpy as np
from utils.augmentations import letterbox
import torch
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes


def detect(image, model):
    pred = model(image, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)
    return pred


def check_face_validity(predictions, image_processed, image_original):
    result_dict = {}
    for i, det in enumerate(predictions):
        det[:, :4] = scale_boxes(image_processed.shape[2:], det[:, :4], image_original.shape).round()

        for face_part in det.cpu().numpy():
            if face_part[-1] not in result_dict:
                result_dict[face_part[-1]] = []
            result_dict[face_part[-1]].append(list(face_part))
    valid_options = {
        0: 2,
        1: 1,
        2: 1
    }

    for face_part, quant in valid_options.items():
        if face_part in result_dict:
            if len(result_dict[face_part]) == valid_options[face_part]:
                pass
            else:
                return False, result_dict
        else:
            return False, result_dict

    return True, result_dict


def read_image(image_path):
    return cv2.imread(image_path)


def prep_image(image, device):
    image = letterbox(image, 640, stride=32, auto=True)[0]  # padded resize
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = np.ascontiguousarray(image)

    image = torch.from_numpy(image).to(device)
    image = image.float()  # uint8 to fp16/32
    image /= 255

    if len(image.shape) == 3:
        image = image[None]
    return image


def prep_model(weights, device, data):
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    image_size = check_img_size((640, 640), s=stride)  # check image size

    model.warmup(imgsz=(1, 3, *image_size))  # warmup
    return model


def main():
    weights = 'weights/real_120.pt'
    device = select_device()
    model = prep_model(weights, device, data='RS_REAL.yaml')  # data used for class names only

    # image_path = 'data/data/Photo/0a9c53710912401e9ad2d47273459ec6.jpeg'
    image_path = '../datasets/3data/1da8964ef3104f96b216a6db31a1dbfc.png'
    image_original = read_image(image_path)
    image_processed = prep_image(image_original, device)

    predictions = detect(image_processed, model)
    validity, results = check_face_validity(predictions, image_processed, image_original)

    print(validity, results)


if __name__ == '__main__':
    main()
