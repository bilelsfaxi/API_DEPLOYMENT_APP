from typing import List
import random
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from . import models, schemas
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

async def get_dog(db: AsyncSession, dog_id: int):
    result = await db.execute(select(models.Dog).filter(models.Dog.id == dog_id))
    return result.scalar_one_or_none()

async def get_dogs(db: AsyncSession, skip: int = 0, limit: int = 10):
    result = await db.execute(select(models.Dog).offset(skip).limit(limit))
    return result.scalars().all()

async def create_dog(db: AsyncSession, dog: schemas.db_schemas.DogCreate):
    db_dog = models.Dog(**dog.dict())
    db.add(db_dog)
    await db.commit()
    await db.refresh(db_dog)
    return db_dog

async def get_session_by_id(db: AsyncSession, session_id: int):
    result = await db.execute(select(models.VideoSession).filter(models.VideoSession.id == session_id))
    return result.scalar_one_or_none()

async def create_video_session(db: AsyncSession, session: schemas.db_schemas.VideoSessionCreate):
    dog = await db.execute(select(models.Dog).filter(models.Dog.id == session.dog_id))
    if not dog.scalar_one_or_none():
        raise HTTPException(status_code=404, detail=f"Dog with id {session.dog_id} not found")
    db_session = models.VideoSession(
        dog_id=session.dog_id,
        posture=session.posture,
        session_start=datetime.now(timezone.utc)
    )
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session)
    return db_session

async def update_session_status(db: AsyncSession, session_id: int):
    session = await get_session_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")
    
    result = await db.execute(
        select(models.PostureDetectionResult).filter(
            models.PostureDetectionResult.session_id == session_id,
            models.PostureDetectionResult.result == "success"
        )
    )
    success_count = len(result.scalars().all())
    
    total_frames = await db.execute(
        select(func.sum(models.PostureDetectionResult.frames_processed)).filter(
            models.PostureDetectionResult.session_id == session_id
        )
    )
    session.total_frames_processed = total_frames.scalar() or 0
    
    if success_count >= 4:
        session.success_detected = True
        session.session_end = datetime.now(timezone.utc)
        validated_posture = models.ValidatedPosture(
            dog_id=session.dog_id,
            posture=session.posture,
            validated_at=datetime.now(timezone.utc)
        )
        db.add(validated_posture)
        await db.commit()
    
    await db.commit()
    await db.refresh(session)
    return success_count

async def create_posture_attempt(db: AsyncSession, attempt: schemas.db_schemas.PostureAttemptCreate):
    session = await get_session_by_id(db, attempt.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session with id {attempt.session_id} not found")
    
    ref_video = await db.execute(
        select(models.ReferencePostureVideo).filter(models.ReferencePostureVideo.id == attempt.video_id)
    )
    ref_video = ref_video.scalar_one_or_none()
    if not ref_video:
        raise HTTPException(status_code=404, detail=f"Reference video with id {attempt.video_id} not found")
    
    if ref_video.posture != session.posture:
        raise HTTPException(status_code=400, detail=f"Video id {attempt.video_id} (posture: {ref_video.posture}) does not match session posture ({session.posture})")
    
    db_attempt = models.PostureDetectionResult(
        session_id=attempt.session_id,
        video_id=attempt.video_id,
        posture=session.posture,
        confidence=attempt.confidence,
        result=attempt.result,
        timestamp=datetime.now(timezone.utc),
        prediction_time=attempt.prediction_time,
        frames_processed=attempt.frames_processed
    )
    db.add(db_attempt)
    await db.commit()
    await db.refresh(db_attempt)
    await update_session_status(db, attempt.session_id)
    return db_attempt

async def get_next_videos_for_session(db: AsyncSession, session_id: int) -> List[schemas.db_schemas.VideoReference]:
    session = await get_session_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")
    
    used_videos_result = await db.execute(
        select(models.PostureDetectionResult.video_id).filter(
            models.PostureDetectionResult.session_id == session_id
        )
    )
    used_video_ids = {row.video_id for row in used_videos_result.scalars().all()}
    
    available_videos_result = await db.execute(
        select(models.ReferencePostureVideo).filter(
            models.ReferencePostureVideo.posture == session.posture
        )
    )
    available_videos = available_videos_result.scalars().all()
    
    if not available_videos:
        raise HTTPException(status_code=404, detail=f"No reference videos found for posture {session.posture}")
    
    not_used_videos = [v for v in available_videos if v.id not in used_video_ids]
    
    selected_videos = not_used_videos[:4]
    if len(selected_videos) < 4:
        additional_videos = random.sample(available_videos, min(4 - len(selected_videos), len(available_videos)))
        selected_videos.extend(additional_videos)
    
    return [
        schemas.db_schemas.VideoReference(id=v.id, posture=v.posture, video_path=v.video_path)
        for v in selected_videos
    ]

async def get_session_status(db: AsyncSession, session_id: int):
    session = await get_session_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")
    
    success_result = await db.execute(
        select(models.PostureDetectionResult).filter(
            models.PostureDetectionResult.session_id == session_id,
            models.PostureDetectionResult.result == "success"
        )
    )
    successful_attempts = len(success_result.scalars().all())
    
    videos_used_result = await db.execute(
        select(models.PostureDetectionResult.video_id).filter(
            models.PostureDetectionResult.session_id == session_id
        )
    )
    videos_used = list({row.video_id for row in videos_used_result.scalars().all()})
    
    return schemas.db_schemas.SessionStatus(
        session_id=session_id,
        posture=session.posture,
        success_detected=session.success_detected,
        successful_attempts=successful_attempts,
        videos_used=videos_used
    )

async def get_validated_postures_by_dog(db: AsyncSession, dog_id: int):
    dog = await get_dog(db, dog_id)
    if not dog:
        raise HTTPException(status_code=404, detail=f"Dog with id {dog_id} not found")
    
    result = await db.execute(
        select(models.ValidatedPosture).filter(models.ValidatedPosture.dog_id == dog_id)
    )
    return result.scalars().all()