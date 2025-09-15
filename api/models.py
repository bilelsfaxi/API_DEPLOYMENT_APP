# api/models.py
import enum
from sqlalchemy import (Column, ForeignKey, Text, Boolean, Float, 
                        Integer, Enum as PgEnum, DateTime)
from sqlalchemy.orm import relationship
# Correction de l'import
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from .database import Base

# Enum Python correspondant au type PostgreSQL
# Mise à jour selon les nouvelles exigences
class PostureEnum(str, enum.Enum):
    assis = "assis"
    debout = "debout"
    a_pieds = "a_pieds"

class Dog(Base):
    __tablename__ = "dogs"

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    breed = Column(Text)
    owner_name = Column(Text)

    sessions = relationship("VideoSession", back_populates="dog")
    validated_postures = relationship("ValidatedPosture", back_populates="dog")

class ReferencePostureVideo(Base):
    __tablename__ = "reference_posture_videos"

    id = Column(Integer, primary_key=True)
    posture = Column(PgEnum(PostureEnum, name="posture_enum", create_type=False), nullable=False)
    video_path = Column(Text, nullable=False)
    description = Column(Text)

class VideoSession(Base):
    __tablename__ = "video_sessions"

    id = Column(Integer, primary_key=True)
    dog_id = Column(Integer, ForeignKey("dogs.id"), nullable=False)

    posture = Column(PgEnum(PostureEnum, name="posture_enum", create_type=False), nullable=False)

    session_start = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    session_end = Column(TIMESTAMP(timezone=True))
    total_frames_processed = Column(Integer, default=0)
    success_detected = Column(Boolean, default=False, nullable=False)

    dog = relationship("Dog", back_populates="sessions")
    results = relationship("PostureDetectionResult", back_populates="session", cascade="all, delete-orphan")

class PostureDetectionResult(Base):
    __tablename__ = "posture_detection_results"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("video_sessions.id"), nullable=False)
    video_id = Column(Integer, ForeignKey("reference_posture_videos.id"), nullable=False)

    posture = Column(PgEnum(PostureEnum, name="posture_enum", create_type=False), nullable=False)
    confidence = Column(Float, nullable=False)
    result = Column(Text, nullable=False) # 'success' or 'fail'
    timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    session = relationship("VideoSession", back_populates="results")
    video = relationship("ReferencePostureVideo")

class ValidatedPosture(Base):
    __tablename__ = "validated_postures"

    id = Column(Integer, primary_key=True)
    dog_id = Column(Integer, ForeignKey("dogs.id"), nullable=False)
    posture = Column(PgEnum(PostureEnum, name="posture_enum", create_type=False), nullable=False)
    validated_at = Column(DateTime(timezone=True), server_default=func.now())

    dog = relationship("Dog", back_populates="validated_postures")

class Embedding(Base):
    __tablename__ = 'embeddings'

    id = Column(Integer, primary_key=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=False)