from sqlalchemy import Column, Integer, Text, Enum, DateTime, Float, ForeignKey, Boolean
from pgvector.sqlalchemy import VECTOR
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import TIMESTAMP
from datetime import datetime
from enum import Enum as PyEnum

Base = declarative_base()

class PostureEnum(PyEnum):
    assis = "assis"
    debout = "debout"
    a_pieds = "a_pieds"

class Dog(Base):
    __tablename__ = "dogs"
    id = Column(Integer, primary_key=True, nullable=False)
    name = Column(Text, nullable=False)
    breed = Column(Text, nullable=True)
    owner_name = Column(Text, nullable=True)

class Embeddings(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(VECTOR(384), nullable=False)

class ReferencePostureVideo(Base):
    __tablename__ = "reference_posture_videos"
    id = Column(Integer, primary_key=True, nullable=False)
    posture = Column(Enum(PostureEnum), nullable=False)
    video_path = Column(Text, nullable=False)
    description = Column(Text, nullable=True)

class VideoSession(Base):
    __tablename__ = "video_sessions"
    id = Column(Integer, primary_key=True, nullable=False)
    dog_id = Column(Integer, ForeignKey("dogs.id"), nullable=False)
    posture = Column(Enum(PostureEnum), nullable=False)
    session_start = Column(TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False)
    session_end = Column(TIMESTAMP(timezone=True), nullable=True)
    total_frames_processed = Column(Integer, server_default='0', nullable=True)
    success_detected = Column(Boolean, server_default=sa.text('false'), nullable=False)

class PostureDetectionResult(Base):
    __tablename__ = "posture_detection_results"
    id = Column(Integer, primary_key=True, nullable=False)
    session_id = Column(Integer, ForeignKey("video_sessions.id"), nullable=False)
    video_id = Column(Integer, ForeignKey("reference_posture_videos.id"), nullable=False)
    posture = Column(Enum(PostureEnum), nullable=False)
    confidence = Column(Float, nullable=False)
    result = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False)
    prediction_time = Column(Float, nullable=True)
    frames_processed = Column(Integer, nullable=True)

class ValidatedPosture(Base):
    __tablename__ = "validated_postures"
    id = Column(Integer, primary_key=True, nullable=False)
    dog_id = Column(Integer, ForeignKey("dogs.id"), nullable=False)
    posture = Column(Enum(PostureEnum), nullable=False)
    validated_at = Column(DateTime(timezone=True), server_default=sa.text('now()'), nullable=True)