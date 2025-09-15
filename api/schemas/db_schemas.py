# api/schemas/db_schemas.py
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import List
from ..models import PostureEnum

# ==================================
# Schémas pour les Chiens (Dog)
# ==================================
class DogBase(BaseModel):
    name: str
    breed: str | None = None
    owner_name: str | None = None

class DogCreate(DogBase):
    pass

class Dog(DogBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

# ==================================
# Schémas pour les Sessions de Validation (VideoSession)
# ==================================
# Modifié pour créer une session basée sur une posture, pas une vidéo
class VideoSessionCreate(BaseModel):
    dog_id: int
    posture: PostureEnum

class VideoSession(BaseModel):
    id: int
    dog_id: int
    posture: PostureEnum
    session_start: datetime
    session_end: datetime | None
    success_detected: bool
    model_config = ConfigDict(from_attributes=True)

# ==================================
# Schémas pour les Tentatives (PostureDetectionResult)
# ==================================
# Nouveau schéma pour enregistrer une tentative
class PostureAttemptCreate(BaseModel):
    session_id: int
    video_id: int
    confidence: float
    result: str # 'success' or 'fail'

class PostureDetectionResult(BaseModel):
    id: int
    session_id: int
    video_id: int
    posture: PostureEnum
    confidence: float
    result: str
    timestamp: datetime
    model_config = ConfigDict(from_attributes=True)

# ==================================
# Schémas pour les Status et la Validation
# ==================================
# Nouveau schéma pour la réponse de l'endpoint de status
class SessionStatus(BaseModel):
    session_id: int
    posture: PostureEnum
    success_detected: bool
    successful_attempts: int
    videos_used: List[int]

# Nouveau schéma pour afficher les postures validées
class ValidatedPosture(BaseModel):
    dog_id: int
    posture: PostureEnum
    validated_at: datetime
    model_config = ConfigDict(from_attributes=True)

# Nouveau schéma pour la réponse de l'endpoint next_video
class VideoReference(BaseModel):
    id: int
    posture: PostureEnum
    video_path: str

    model_config = ConfigDict(from_attributes=True)