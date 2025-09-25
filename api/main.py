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
# Le chemin du dossier 'static' est relatif à la racine du projet, pas au dossier 'api'.
static_dir = "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Configuration des templates HTML (Jinja2) - Le chemin est maintenant géré dans le routeur UI
# pour une meilleure modularité.
api_root = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(api_root, "routers")

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
app.include_router(ui_router.create_router(templates_directory=templates_dir))

@app.on_event("startup")
async def startup_event():
    """Code exécuté au démarrage de l'application."""
    # Le seeding des données de référence est maintenant géré par la migration Alembic.
    # Cette fonction est maintenant beaucoup plus rapide.
    logger.info("✅ Application démarrée avec succès.")

@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def root():
    """
    Redirige la racine de l'API vers la documentation interactive.
    """
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)