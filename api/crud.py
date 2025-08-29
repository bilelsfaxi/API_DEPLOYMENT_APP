# api/crud.py
import random
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_
from fastapi import HTTPException

from . import models
from .schemas import db_schemas

# Configuration du logging
logger = logging.getLogger(__name__)

# ==================================
# Fonctions CRUD pour les Chiens (Dog)
# ==================================
async def get_dog(db: AsyncSession, dog_id: int):
    result = await db.execute(select(models.Dog).filter(models.Dog.id == dog_id))
    return result.scalar_one_or_none()

async def get_dogs(db: AsyncSession, skip: int = 0, limit: int = 10):
    result = await db.execute(select(models.Dog).offset(skip).limit(limit))
    return result.scalars().all()

async def create_dog(db: AsyncSession, dog: db_schemas.DogCreate):
    logger.info(f"Creating new dog with name: {dog.name}")
    db_dog = models.Dog(**dog.model_dump())
    db.add(db_dog)
    await db.commit()
    await db.refresh(db_dog)
    return db_dog

# ==================================
# Fonctions CRUD pour les Sessions (VideoSession)
# ==================================
async def get_session_by_id(db: AsyncSession, session_id: int):
    result = await db.execute(select(models.VideoSession).filter(models.VideoSession.id == session_id))
    return result.scalar_one_or_none()

async def create_video_session(db: AsyncSession, session: db_schemas.VideoSessionCreate):
    logger.info(f"Creating new session for dog_id={session.dog_id} and posture='{session.posture.value}'")
    dog = await get_dog(db, session.dog_id)
    if not dog:
        logger.warning(f"Attempted to create session for non-existent dog_id: {session.dog_id}")
        raise HTTPException(status_code=404, detail=f"Dog with id {session.dog_id} not found")

    db_session = models.VideoSession(**session.model_dump())
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session)
    logger.info(f"Successfully created session with id: {db_session.id}")
    return db_session

# ==================================
# Logique métier pour la validation des postures
# ==================================
async def update_session_status(db: AsyncSession, session_id: int):
    """Vérifie si une session atteint 4 succès et met à jour son statut."""
    session = await get_session_by_id(db, session_id)
    if not session or session.success_detected:
        return session # Ne rien faire si la session n'existe pas ou est déjà validée

    query = select(func.count()).select_from(models.PostureDetectionResult).where(
        and_(
            models.PostureDetectionResult.session_id == session_id,
            models.PostureDetectionResult.result == 'success'
        )
    )
    success_count = await db.scalar(query)

    if success_count >= 4:
        logger.info(f"Session {session_id} has reached {success_count} successes. Validating posture '{session.posture.value}' for dog {session.dog_id}.")
        session.success_detected = True
        db.add(session)

        validated_posture = models.ValidatedPosture(dog_id=session.dog_id, posture=session.posture)
        db.add(validated_posture)

        await db.commit()
        await db.refresh(session)
        logger.info(f"Session {session_id} successfully validated.")
    return session

async def create_posture_attempt(db: AsyncSession, attempt: db_schemas.PostureAttemptCreate):
    """Enregistre une tentative de posture et vérifie si la session est validée."""
    session = await get_session_by_id(db, attempt.session_id)
    if not session:
        logger.warning(f"Posture attempt failed: Session with id {attempt.session_id} not found.")
        raise HTTPException(status_code=404, detail=f"Session with id {attempt.session_id} not found")

    ref_video_result = await db.execute(select(models.ReferencePostureVideo).filter(models.ReferencePostureVideo.id == attempt.video_id))
    ref_video = ref_video_result.scalar_one_or_none()

    if not ref_video:
        logger.warning(f"Posture attempt failed: Reference video with id {attempt.video_id} not found.")
        raise HTTPException(status_code=404, detail=f"Reference video with id {attempt.video_id} not found")

    if ref_video.posture != session.posture:
        logger.warning(f"Posture mismatch in session {session.id}: Video posture '{ref_video.posture.value}' != Session posture '{session.posture.value}'.")
        raise HTTPException(status_code=400, detail=f"Video id {attempt.video_id} (posture: {ref_video.posture}) does not match session posture ({session.posture})")

    db_attempt = models.PostureDetectionResult(
        session_id=attempt.session_id,
        video_id=attempt.video_id,
        posture=session.posture,
        confidence=attempt.confidence,
        result=attempt.result
    )
    db.add(db_attempt)
    await db.commit()
    await db.refresh(db_attempt)
    logger.info(f"Recorded new posture attempt {db_attempt.id} for session {db_attempt.session_id} with result '{db_attempt.result}'.")

    await update_session_status(db, attempt.session_id)
    return db_attempt

async def get_next_video_for_session(db: AsyncSession, session_id: int):
    """Sélectionne une vidéo de référence non utilisée pour la prochaine tentative."""
    session = await get_session_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")

    # 1. Trouver les vidéos déjà utilisées dans cette session
    used_videos_query = select(models.PostureDetectionResult.video_id).where(models.PostureDetectionResult.session_id == session_id)
    used_videos_result = await db.execute(used_videos_query)
    used_video_ids = used_videos_result.scalars().all()

    # 2. Trouver toutes les vidéos disponibles pour la posture de la session
    available_videos_query = select(models.ReferencePostureVideo).where(
        models.ReferencePostureVideo.posture == session.posture
    )
    available_videos_result = await db.execute(available_videos_query)
    all_posture_videos = available_videos_result.scalars().all()

    if not all_posture_videos:
        raise HTTPException(status_code=404, detail=f"No reference videos found for posture {session.posture}")

    # 3. Filtrer pour trouver les vidéos non encore utilisées
    unused_videos = [v for v in all_posture_videos if v.id not in used_video_ids]

    # 4. S'il n'y a plus de vidéo non utilisée, choisir au hasard parmi toutes les vidéos disponibles.
    #    Sinon, choisir au hasard parmi les non-utilisées.
    if not unused_videos:
        logger.info(f"All 4 videos have been used for session {session_id}. Selecting randomly from all available.")
        return random.choice(all_posture_videos)
    else:
        logger.info(f"{len(unused_videos)} unused videos remaining for session {session_id}. Selecting one.")
        return random.choice(unused_videos)


async def get_session_status(db: AsyncSession, session_id: int):
    """Récupère le statut complet d'une session."""
    session = await get_session_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")

    success_query = select(func.count()).select_from(models.PostureDetectionResult).where(
        and_(models.PostureDetectionResult.session_id == session_id, models.PostureDetectionResult.result == 'success')
    )
    success_count = await db.scalar(success_query)

    videos_query = select(models.PostureDetectionResult.video_id).where(models.PostureDetectionResult.session_id == session_id).distinct()
    videos_result = await db.execute(videos_query)
    videos_used = videos_result.scalars().all()

    return db_schemas.SessionStatus(
        session_id=session.id,
        posture=session.posture,
        success_detected=session.success_detected,
        successful_attempts=success_count,
        videos_used=videos_used
    )

async def get_validated_postures_by_dog(db: AsyncSession, dog_id: int):
    """Récupère toutes les postures validées pour un chien donné."""
    dog = await get_dog(db, dog_id)
    if not dog:
        raise HTTPException(status_code=404, detail=f"Dog with id {dog_id} not found")

    query = select(models.ValidatedPosture).where(models.ValidatedPosture.dog_id == dog_id)
    result = await db.execute(query)
    return result.scalars().all()
