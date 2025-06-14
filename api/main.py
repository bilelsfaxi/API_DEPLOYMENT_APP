from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from websockets.exceptions import ConnectionClosed
import os
import logging
import cv2
import numpy as np
import asyncio

# Importation des modules internes
from api.detectors import detectors_yolo11
from api.routers import routers_yolo11

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(title="YOLOv11 Dog Posture Detection API")

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

@app.on_event("startup")
async def startup_event():
    """Code exécuté au démarrage de l'application."""
    logger.info("✅ Application démarrée avec succès.")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Route racine pour afficher l'interface web."""
    return templates.TemplateResponse("index.html", {"request": request})
