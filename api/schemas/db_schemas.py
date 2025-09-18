from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from ..models import PostureEnum

class DogBase(BaseModel):
    name: str
    breed: Optional[str] = None
    owner_name: Optional[str] = None

class DogCreate(DogBase):
    pass

class Dog(DogBase):
    id: int
    class Config:
        from_attributes = True

class VideoSessionCreate(BaseModel):
    dog_id: int
    posture: PostureEnum

class VideoSession(BaseModel):
    id: int
    dog_id: int
    posture: PostureEnum
    session_start: datetime
    session_end: Optional[datetime] = None
    total_frames_processed: Optional[int] = None
    success_detected: bool = False
    class Config:
        from_attributes = True

class PostureAttemptCreate(BaseModel):
    session_id: int
    video_id: int
    confidence: float
    result: str
    prediction_time: float
    frames_processed: int

class PostureDetectionResult(BaseModel):
    id: int
    session_id: int
    video_id: int
    posture: PostureEnum
    confidence: float
    result: str
    timestamp: datetime
    prediction_time: float
    frames_processed: int
    class Config:
        from_attributes = True

class SessionStatus(BaseModel):
    session_id: int
    posture: PostureEnum
    success_detected: bool
    successful_attempts: int
    videos_used: List[int]

class ValidatedPosture(BaseModel):
    id: int
    dog_id: int
    posture: PostureEnum
    validated_at: datetime
    class Config:
        from_attributes = True

class VideoReference(BaseModel):
    id: int
    posture: PostureEnum
    video_path: str
    class Config:
        from_attributes = True