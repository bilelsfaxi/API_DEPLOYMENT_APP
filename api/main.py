from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends, HTTPException, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import os
import logging
import cv2
import numpy as np
import asyncio

# Importation des modules internes
from api.detectors import detectors_yolo11
from api.routers import routers_yolo11, db_router, ui_router
from api.database import engine, Base, get_db
from api import crud, models

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(title="YOLOv11 Dog Posture Detection API")

# Monter le répertoire statique pour servir les vidéos de référence et le CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration des templates HTML (Jinja2)
templates = Jinja2Templates(directory="api/templates")

# Chemin absolu vers le modèle YOLOv11
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "final_model_yolo11.pt")

# Initialisation du détecteur avec le modèle
detector = detectors_yolo11.YOLOv11Detector(model_path=MODEL_PATH)

# Injection du détecteur dans le routeur
routers_yolo11.detector = detector

# Enregistrement des routes du routeur YOLOv11
app.include_router(routers_yolo11.router)

# Enregistrement des routes du routeur de la base de données
app.include_router(db_router.router)

# Enregistrement des routes du routeur de l'interface utilisateur
app.include_router(ui_router.router)

async def seed_reference_videos(db: AsyncSession):
    """
    Remplit la base de données avec les vidéos de référence du dossier static/videos.
    Ne fait rien si les vidéos existent déjà pour éviter les doublons.
    Normalise les noms de fichiers pour éviter les espaces inutiles.
    """
    from api.models import ReferencePostureVideo, PostureEnum

    STATIC_VIDEOS_DIR = "static/videos"
    if not os.path.isdir(STATIC_VIDEOS_DIR):
        logger.warning(f"Le dossier des vidéos de référence '{STATIC_VIDEOS_DIR}' n'a pas été trouvé. Le seeding est ignoré.")
        return

    try:
        for filename in os.listdir(STATIC_VIDEOS_DIR):
            if filename.endswith(".mp4"):
                try:
                    # Normaliser le nom du fichier : supprimer les espaces de début/fin,
                    # puis remplacer les espaces internes par des underscores.
                    normalized_filename = " ".join(filename.strip().split()).replace(" ", "_")
                    base_name = normalized_filename.removesuffix('.mp4').strip()
                    posture_part = base_name.rsplit('_', 1)[0].strip()
                    posture_name = posture_part.replace("chien_", "").replace("chiens_", "").strip()
                    posture_enum = PostureEnum(posture_name)

                    # Vérifier si la vidéo existe déjà pour éviter les doublons
                    video_path = f"static/videos/{normalized_filename}"
                    result = await db.execute(select(ReferencePostureVideo).filter_by(video_path=video_path))
                    if result.scalar_one_or_none() is None:
                        new_video = ReferencePostureVideo(posture=posture_enum, video_path=video_path)
                        db.add(new_video)
                        logger.info(f"Ajout de la vidéo de référence à la BDD : {video_path}")
                except (ValueError, IndexError):
                    logger.warning(f"Impossible de traiter le fichier vidéo '{filename}'. Le nom ne correspond pas au format attendu (ex: 'assis_1.mp4').")
        await db.commit()
    except Exception as e:
        logger.error(f"Erreur lors du seeding de la base de données : {e}")
        await db.rollback()

@app.on_event("startup")
async def startup_event():
    """Code exécuté au démarrage de l'application."""
    from api.database import SessionLocal
    try:
        async with SessionLocal() as db:
            await seed_reference_videos(db)
        logger.info("✅ Application démarrée avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors du démarrage de l'application : {str(e)}")
        raise

@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def root():
    """
    Redirige la racine de l'API vers la documentation interactive.
    """
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)