from fastapi import APIRouter, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.responses import Response, JSONResponse, FileResponse
from api.schemas.schemas_yolo11 import DetectionResponse, VideoDetectionResponse, OutputFormat, Detection
from api.detectors.detectors_yolo11 import YOLOv11Detector
from api import crud, schemas, database
from sqlalchemy.ext.asyncio import AsyncSession
import os
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import asyncio
import base64
import logging
import time

router = APIRouter(prefix="/yolo", tags=["YOLOv11"])
detector = None  # Le détecteur sera injecté depuis main.py
video_source = None
streaming_active = False

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    session_id: int = Query(...),
    video_id: int = Query(...),
    output_format: OutputFormat = Query(OutputFormat.IMAGE, description="Format de sortie: image, json, ou csv"),
    db: AsyncSession = Depends(database.get_db)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        detections, metrics = detector.process_image(image_np)
        
        if detections:
            attempt = schemas.PostureAttemptCreate(
                session_id=session_id,
                video_id=video_id,
                confidence=metrics["avg_confidence"],
                result=detections[0]["result"],
                prediction_time=metrics["prediction_time"],
                frames_processed=metrics["frames_processed"]
            )
            await crud.create_posture_attempt(db, attempt)

        if output_format == OutputFormat.IMAGE:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_output:
                temp_output_path = temp_output.name
                detector.process_image(image_np, temp_output_path)
            with open(temp_output_path, "rb") as annotated_file:
                encoded_image = annotated_file.read()
            os.remove(temp_output_path)
            return Response(content=encoded_image, media_type="image/jpeg")
        
        elif output_format == OutputFormat.JSON:
            detection_objects = [
                Detection(
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    bbox=det["bbox"]
                ) for det in detections
            ]
            return DetectionResponse(
                detections=detection_objects,
                total_detections=len(detections),
                prediction_time=metrics["prediction_time"],
                avg_confidence=metrics["avg_confidence"],
                frames_processed=metrics["frames_processed"]
            )
            
        elif output_format == OutputFormat.CSV:
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
    session_id: int = Query(...),
    video_id: int = Query(...),
    db: AsyncSession = Depends(database.get_db)
):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(contents)
            temp_input_path = temp_input.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
            temp_output_path = temp_output.name
            result = detector.process_video(temp_input_path, temp_output_path)

        if result["detections"]:
            attempt = schemas.PostureAttemptCreate(
                session_id=session_id,
                video_id=video_id,
                confidence=result["avg_confidence"],
                result=result["detections"][0]["result"],
                prediction_time=result["prediction_time"],
                frames_processed=result["frames_processed"]
            )
            await crud.create_posture_attempt(db, attempt) # On ne récupère pas le retour

        with open(temp_output_path, "rb") as output_file:
            encoded_video = output_file.read()

        os.remove(temp_input_path)
        os.remove(temp_output_path)

        return Response(content=encoded_video, media_type="video/mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de la vidéo : {str(e)}")

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

@router.websocket("/ws/{session_id}/{video_id}")
async def stream_video(websocket: WebSocket, session_id: int, video_id: int, db: AsyncSession = Depends(database.get_db)):
    global video_source, streaming_active
    await websocket.accept()
    if not streaming_active or not video_source or not video_source.isOpened():
        await websocket.close()
        return

    try:
        frame_count = 0
        confidences = []
        start_time = time.time()
        while streaming_active:
            ret, frame = video_source.read()
            if not ret:
                break
            frame_count += 1
            detections, metrics = detector.process_image(frame)
            for det in detections:
                confidences.append(det["confidence"])
            
            if detections:
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                attempt = schemas.PostureAttemptCreate(
                    session_id=session_id,
                    video_id=video_id,
                    confidence=avg_confidence,
                    result=detections[0]["result"],
                    prediction_time=metrics["prediction_time"],
                    frames_processed=metrics["frames_processed"]
                )
                await crud.create_posture_attempt(db, attempt)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_json({"frame": frame_base64, "detections": detections})
            await asyncio.sleep(0.1)
        
        prediction_time = time.time() - start_time
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        print(f"WebSocket stream: {frame_count} frames, {prediction_time:.2f}s, avg confidence: {avg_confidence:.2f}")

    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()