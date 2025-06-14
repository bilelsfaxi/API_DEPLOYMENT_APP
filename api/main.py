from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import logging

# Importation des modules internes
from api.detectors import detectors_yolo11
from api.routers import routers_yolo11

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App FastAPI
app = FastAPI(title="YOLOv11 Dog Posture Detection API")

# Templates HTML
templates = Jinja2Templates(directory="templates")

# Modèle YOLOv11
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "final_model_yolo11.pt")
detector = detectors_yolo11.YOLOv11Detector(model_path=MODEL_PATH)
routers_yolo11.detector = detector
app.include_router(routers_yolo11.router)

@app.on_event("startup")
async def startup_event():
    logger.info("✅ Application démarrée avec succès.")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
