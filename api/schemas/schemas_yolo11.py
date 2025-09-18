from pydantic import BaseModel
from typing import List
from enum import Enum

class OutputFormat(str, Enum):
    IMAGE = "image"
    JSON = "json"
    CSV = "csv"

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]

class DetectionResponse(BaseModel):
    detections: List[Detection]
    total_detections: int
    prediction_time: float
    avg_confidence: float
    frames_processed: int

class VideoDetectionResponse(BaseModel):
    detections: List[Detection]
    prediction_time: float
    avg_confidence: float
    frames_processed: int