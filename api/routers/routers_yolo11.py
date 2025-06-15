from fastapi import APIRouter, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse, FileResponse
from api.schemas.schemas_yolo11 import DetectionResponse
import os
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import asyncio

router = APIRouter(prefix="/yolo", tags=["YOLOv11"])
detector = None  # Injecté dynamiquement depuis main.py

video_source = None
streaming_active = False


@router.post("/predict", response_model=DetectionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")

    if detector is None:
        raise HTTPException(status_code=500, detail="Modèle non initialisé")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_output:
            temp_output_path = temp_output.name
            detections = detector.process_image(image_np, temp_output_path)

        with open(temp_output_path, "rb") as annotated_file:
            encoded_image = annotated_file.read()

        os.remove(temp_output_path)

        return Response(content=encoded_image, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur traitement image : {str(e)}")


@router.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une vidéo")

    if detector is None:
        raise HTTPException(status_code=500, detail="Modèle non initialisé")

    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(contents)
            temp_input_path = temp_input.name

        os.makedirs("temp_results", exist_ok=True)
        output_path = os.path.join("temp_results", f"video_annotated_{os.path.basename(temp_input_path)}")

        # Appel du traitement vidéo avec journalisation
        detections = detector.process_video(temp_input_path, output_path)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="La vidéo annotée n’a pas été générée.")

        os.remove(temp_input_path)

        return FileResponse(
            path=output_path,
            filename="video_annotated.mp4",
            media_type="video/mp4"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur traitement vidéo : {str(e)}")


@router.post("/start-stream")
async def start_stream():
    global video_source, streaming_active

    if streaming_active:
        return JSONResponse(content={"message": "Streaming déjà actif"})

    video_source = cv2.VideoCapture(0)
    if not video_source.isOpened():
        return JSONResponse(status_code=500, content={"message": "Webcam inaccessible"})

    streaming_active = True
    return JSONResponse(content={"message": "Streaming activé"})


@router.post("/stop-stream")
async def stop_stream():
    global video_source, streaming_active

    if streaming_active:
        if video_source:
            video_source.release()
        streaming_active = False
        return JSONResponse(content={"message": "Streaming arrêté"})

    return JSONResponse(content={"message": "Streaming déjà arrêté"})


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

            # Vérification du détecteur
            if detector:
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
