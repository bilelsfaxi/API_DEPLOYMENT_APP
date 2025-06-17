from fastapi import APIRouter, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Form, Query
from fastapi.responses import Response, JSONResponse, FileResponse
from api.schemas.schemas_yolo11 import DetectionResponse, VideoDetectionResponse, OutputFormat, Detection
import os
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import asyncio
from typing import Optional

router = APIRouter(prefix="/yolo", tags=["YOLOv11"])
detector = None

video_source = None
streaming_active = False

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    output_format: OutputFormat = Query(OutputFormat.IMAGE, description="Format de sortie: image, json, ou csv")
):
    """
    Prédiction sur une image avec choix du format de sortie
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        if detector is None:
            raise RuntimeError("Modèle non initialisé")

        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        # Traitement de l'image
        if output_format == OutputFormat.IMAGE:
            # Mode original - retourne l'image annotée
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_output:
                temp_output_path = temp_output.name
                detections = detector.process_image(image_np, temp_output_path)

            with open(temp_output_path, "rb") as annotated_file:
                encoded_image = annotated_file.read()

            os.remove(temp_output_path)
            return Response(content=encoded_image, media_type="image/jpeg")
        
        else:
            # Mode JSON ou CSV - retourne les données de détection
            detections = detector.process_image(image_np)
            
            if output_format == OutputFormat.JSON:
                detection_objects = [
                    Detection(
                        class_name=det["class_name"],
                        confidence=det["confidence"],
                        bbox=det["bbox"]
                    ) for det in detections
                ]
                
                response = DetectionResponse(
                    detections=detection_objects,
                    total_detections=len(detections)
                )
                return response
            
            elif output_format == OutputFormat.CSV:
                # Créer un fichier CSV temporaire
                os.makedirs("temp_results", exist_ok=True)
                csv_filename = f"detections_{file.filename.split('.')[0]}.csv"
                csv_path = os.path.join("temp_results", csv_filename)
                
                detector.save_detections_to_csv(detections, csv_path)
                
                return FileResponse(
                    path=csv_path,
                    filename=csv_filename,
                    media_type="text/csv"
                )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

@router.post("/predict-video")
async def predict_video(
    file: UploadFile = File(...),
    output_format: OutputFormat = Query(OutputFormat.IMAGE, description="Format de sortie: image (vidéo annotée), json, ou csv")
):
    """
    Prédiction sur une vidéo avec choix du format de sortie
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    try:
        if detector is None:
            raise RuntimeError("Modèle non initialisé")

        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(contents)
            temp_input_path = temp_input.name

        os.makedirs("temp_results", exist_ok=True)
        
        if output_format == OutputFormat.IMAGE:
            # Mode original - retourne la vidéo annotée
            output_path = os.path.join("temp_results", f"video_annotated_{os.path.basename(temp_input_path)}")
            detector.process_video(temp_input_path, output_path)
            
            os.remove(temp_input_path)
            
            return FileResponse(
                path=output_path,
                filename="video_annotated.mp4", 
                media_type="video/mp4"
            )
        
        else:
            # Mode JSON ou CSV - traitement sans sauvegarde de vidéo annotée
            video_results = detector.process_video(temp_input_path)
            detections = video_results['detections']
            
            os.remove(temp_input_path)
            
            if output_format == OutputFormat.JSON:
                detection_objects = [
                    Detection(
                        class_name=det["class_name"],
                        confidence=det["confidence"],
                        bbox=det["bbox"]
                    ) for det in detections
                ]
                
                response = VideoDetectionResponse(
                    detections=detection_objects,
                    total_detections=len(detections),
                    total_frames=video_results['total_frames'],
                    video_duration=video_results['duration']
                )
                return response
            
            elif output_format == OutputFormat.CSV:
                csv_filename = f"video_detections_{file.filename.split('.')[0]}.csv"
                csv_path = os.path.join("temp_results", csv_filename)
                
                detector.save_detections_to_csv(detections, csv_path)
                
                return FileResponse(
                    path=csv_path,
                    filename=csv_filename,
                    media_type="text/csv"
                )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur traitement vidéo : {str(e)}")

# Routes existantes inchangées
@router.post("/start-stream")
async def start_stream():
    global video_source, streaming_active
    if not streaming_active:
        video_source = cv2.VideoCapture(0)
        if not video_source.isOpened():
            return JSONResponse(status_code=500, content={"message": "Webcam inaccessible"})
        streaming_active = True
        return JSONResponse(content={"message": "Streaming activé"})
    return JSONResponse(content={"message": "Streaming déjà actif"})

@router.post("/stop-stream")
async def stop_stream():
    global video_source, streaming_active
    if streaming_active:
        if video_source:
            video_source.release()
        streaming_active = False
        return JSONResponse(content={"message": "Streaming arrêté"})
    return JSONResponse(content={"message": "Déjà arrêté"})

@router.websocket("/ws")
async def stream_video(websocket: WebSocket):
    global video_source, streaming_active
    await websocket.accept()
    try:
        while True:
            if not streaming_active or not video_source or not video_source.isOpened():
                await websocket.send_text("Streaming désactivé")
                await asyncio.sleep(1)
                continue
            success, frame = video_source.read()
            if not success:
                streaming_active = False
                await websocket.send_text("Streaming arrêté")
                break

            frame = cv2.resize(frame, (640, 360))
            detections = detector.process_image(frame)
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                text_y = max(y1 - 10, 20)
                cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.03)

    except (WebSocketDisconnect, Exception):
        print("Client déconnecté")
    finally:
        if video_source:
            video_source.release()
        streaming_active = False