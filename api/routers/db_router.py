from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from .. import crud
from ..schemas import db_schemas
from ..database import get_db

router = APIRouter(prefix="/db", tags=["Database"])

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

@router.post("/sessions/", response_model=db_schemas.VideoSession, status_code=201, summary="Créer une nouvelle session de validation")
async def create_session_endpoint(session: db_schemas.VideoSessionCreate, db: AsyncSession = Depends(get_db)):
    return await crud.create_video_session(db=db, session=session)

@router.post("/posture_attempts/", response_model=db_schemas.PostureDetectionResult, status_code=201, summary="Enregistrer une tentative de posture")
async def create_posture_attempt_endpoint(attempt: db_schemas.PostureAttemptCreate, db: AsyncSession = Depends(get_db)):
    return await crud.create_posture_attempt(db=db, attempt=attempt)

@router.get("/sessions/{session_id}/status", response_model=db_schemas.SessionStatus, summary="Vérifier le statut d'une session")
async def get_session_status_endpoint(session_id: int, db: AsyncSession = Depends(get_db)):
    return await crud.get_session_status(db=db, session_id=session_id)

@router.get("/sessions/{session_id}/next_videos", response_model=List[db_schemas.VideoReference], summary="Obtenir les prochaines vidéos de référence")
async def get_next_videos_endpoint(session_id: int, db: AsyncSession = Depends(get_db)):
    return await crud.get_next_videos_for_session(db=db, session_id=session_id)

@router.get("/dogs/{dog_id}/validated_postures", response_model=List[db_schemas.ValidatedPosture], summary="Lister les postures validées pour un chien")
async def get_validated_postures_endpoint(dog_id: int, db: AsyncSession = Depends(get_db)):
    return await crud.get_validated_postures_by_dog(db=db, dog_id=dog_id)