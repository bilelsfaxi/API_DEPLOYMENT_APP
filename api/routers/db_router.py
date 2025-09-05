# api/routers/db_router.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from .. import crud
from ..schemas import db_schemas
from ..database import SessionLocal
from ..models import ReferencePostureVideo

router = APIRouter(prefix="/db", tags=["Database"])

# Dépendance pour obtenir la session de base de données
async def get_db():
    async with SessionLocal() as session:
        yield session

# ==================================
# Endpoints pour les Chiens (Dog) - Inchangé
# ==================================
@router.post("/dogs/", response_model=db_schemas.Dog, status_code=201, summary="Créer un nouveau chien")
async def create_dog_endpoint(dog: db_schemas.DogCreate, db: AsyncSession = Depends(get_db)):
    return await crud.create_dog(db=db, dog=dog)

@router.get("/dogs/", response_model=List[db_schemas.Dog], summary="Lister les chiens")
async def read_dogs_endpoint(skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)):
    return await crud.get_dogs(db, skip=skip, limit=limit)

@router.get("/dogs/{dog_id}", response_model=db_schemas.Dog, summary="Obtenir un chien par ID")
async def read_dog_endpoint(dog_id: int, db: AsyncSession = Depends(get_db)):
    db_dog = await crud.get_dog(db, dog_id=dog_id)
    if db_dog is None:
        raise HTTPException(status_code=404, detail="Dog not found")
    return db_dog

# ==================================
# Endpoints pour la Validation des Postures - NOUVEAU
# ==================================
@router.post("/sessions/", response_model=db_schemas.VideoSession, status_code=201, summary="Créer une nouvelle session de validation")
async def create_session_endpoint(session: db_schemas.VideoSessionCreate, db: AsyncSession = Depends(get_db)):
    """Crée une nouvelle session de validation pour un chien et une posture spécifique."""
    return await crud.create_video_session(db=db, session=session)

@router.post("/posture_attempts/", response_model=db_schemas.PostureDetectionResult, status_code=201, summary="Enregistrer une tentative de posture")
async def create_posture_attempt_endpoint(attempt: db_schemas.PostureAttemptCreate, db: AsyncSession = Depends(get_db)):
    """
    Enregistre le résultat d'une tentative de posture.
    Cette opération déclenche la logique de validation (règle des 4 succès).
    """
    return await crud.create_posture_attempt(db=db, attempt=attempt)

@router.get("/sessions/{session_id}/status", response_model=db_schemas.SessionStatus, summary="Vérifier le statut d'une session")
async def get_session_status_endpoint(session_id: int, db: AsyncSession = Depends(get_db)):
    """Récupère le statut complet d'une session, incluant le nombre de succès."""
    return await crud.get_session_status(db=db, session_id=session_id)

@router.get("/sessions/{session_id}/next_video", response_model=db_schemas.PostureDetectionResult, summary="Obtenir la prochaine vidéo de référence")
async def get_next_video_endpoint(session_id: int, db: AsyncSession = Depends(get_db)):
    """Suggère une vidéo de référence pour la prochaine tentative de la session."""
    # Note: The response_model is a bit of a stretch, but it contains the video info.
    # A dedicated schema could be better in the long run.
    next_video = await crud.get_next_video_for_session(db=db, session_id=session_id)
    # We need to shape the response to fit PostureDetectionResult or a new schema
    # This is a simplified response for now. A proper implementation might need a dedicated schema.
    return {
        "id": next_video.id,
        "session_id": session_id,
        "video_id": next_video.id,
        "posture": next_video.posture,
        "confidence": 1.0, # Placeholder
        "result": "suggestion", # Placeholder
        "timestamp": "2025-01-01T00:00:00Z" # Placeholder
    }


@router.get("/dogs/{dog_id}/validated_postures", response_model=List[db_schemas.ValidatedPosture], summary="Lister les postures validées pour un chien")
async def get_validated_postures_endpoint(dog_id: int, db: AsyncSession = Depends(get_db)):
    """Récupère la liste de toutes les postures qui ont été validées pour un chien."""
    return await crud.get_validated_postures_by_dog(db=db, dog_id=dog_id)