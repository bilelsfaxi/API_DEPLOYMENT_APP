from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class OutputFormat(str, Enum):
    IMAGE = "image"
    JSON = "json" 
    CSV = "csv"

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class DetectionResponse(BaseModel):
    detections: List[Detection]
    total_detections: int
    
class VideoDetectionResponse(BaseModel):
    detections: List[Detection]
    total_detections: int
    total_frames: int
    video_duration: Optional[float] = None