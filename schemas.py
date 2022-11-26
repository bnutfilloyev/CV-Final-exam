from pydantic import BaseModel


class PostEyeDetect(BaseModel):
    image: str
    blink_threshold: float = 0.5


class PostOcclusionDetect(BaseModel):
    image: str


class EyeResult(BaseModel):
    left_eye: float
    right_eye: float


class EyeDetectResponse(BaseModel):
    is_blinking: bool
    results: EyeResult


class OcclusionDetectResponse(BaseModel):
    is_occluded: bool
