import logging

from fastapi import FastAPI

from schemas import *
from detection import base64_2_bytestr, detector

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()


@app.post("/eye", response_model=EyeDetectResponse)
def eye(image: PostEyeDetect):
    image_str = image.image
    byte_image = base64_2_bytestr(image_str)
    return detector.eye_detect(byte_image)


@app.post("/occlusion", response_model=OcclusionDetectResponse)
def occlusion(image: PostOcclusionDetect):
    image_str = image.image
    byte_image = base64_2_bytestr(image_str)
    return detector.occlusion_detect(byte_image)
