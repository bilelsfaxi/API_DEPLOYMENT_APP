from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import os
import logging
import cv2
import numpy as np
import asyncio
from sqlalchemy.future import select
# Importation des modules internes
from api.detectors import detectors_yolo11
from api.routers import routers_yolo11, db_router
from api.database import engine, Base

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(title="YOLOv11 Dog Posture Detection API")

# Monter le répertoire statique pour servir les vidéos de référence
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration des templates HTML (Jinja2)
templates = Jinja2Templates(directory="templates")

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

async def seed_reference_videos():
    """
    Remplit la base de données avec les vidéos de référence du dossier static/videos.
    Ne fait rien si les vidéos existent déjà pour éviter les doublons.
    """
    from api.database import SessionLocal
    from api.models import ReferencePostureVideo, PostureEnum
    from sqlalchemy.future import select

    STATIC_VIDEOS_DIR = "static/videos"
    if not os.path.isdir(STATIC_VIDEOS_DIR):
        logger.warning(f"Le dossier des vidéos de référence '{STATIC_VIDEOS_DIR}' n'a pas été trouvé. Le seeding est ignoré.")
        return

    async with SessionLocal() as db:
        try:
            for filename in os.listdir(STATIC_VIDEOS_DIR):
                if filename.endswith(".mp4"):
                    try:
                        # Logique d'extraction de posture améliorée
                        # 1. Enlever l'extension .mp4
                        # 2. Séparer le nom de la posture du numéro (ex: "chien a pieds_1" -> ["chien a pieds", "1"])
                        # 3. Remplacer "a pieds" par "a_pieds" pour correspondre à l'enum
                        base_name = filename.removesuffix('.mp4').strip()
                        posture_part = base_name.rsplit('_', 1)[0].strip()
                        # Nettoie le préfixe "chien" ou "chiens" et remplace l'espace par un underscore
                        posture_name = posture_part.replace("chien ", "").replace("chiens ", "").strip().replace(" ", "_")
                        posture_enum = PostureEnum(posture_name)

                        # Vérifier si la vidéo existe déjà pour éviter les doublons
                        result = await db.execute(select(ReferencePostureVideo).filter_by(video_path=filename))
                        if result.scalar_one_or_none() is None:
                            new_video = ReferencePostureVideo(posture=posture_enum, video_path=filename)
                            db.add(new_video)
                            logger.info(f"Ajout de la vidéo de référence à la BDD : {filename}")
                    except (ValueError, IndexError):
                        logger.warning(f"Impossible de traiter le fichier vidéo '{filename}'. Le nom ne correspond pas au format attendu (ex: 'assis_1.mp4').")
            await db.commit()
        except Exception as e:
            logger.error(f"Erreur lors du seeding de la base de données : {e}")
            await db.rollback()

@app.on_event("startup")
async def startup_event():
    """Code exécuté au démarrage de l'application."""
    # La création des tables est maintenant gérée par Alembic.
    # Laisser cette ligne peut causer des conflits ou des incohérences.
    # Il est recommandé de lancer `alembic upgrade head` avant de démarrer l'application.
    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)

    await seed_reference_videos()

    logger.info("✅ Application démarrée avec succès.")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Route racine pour afficher l'interface web."""
    return templates.TemplateResponse("index.html", {"request": request})