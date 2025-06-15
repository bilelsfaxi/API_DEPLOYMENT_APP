from ultralytics import YOLO
import os
import cv2
import numpy as np
from typing import List, Dict

class YOLOv11Detector:
    def __init__(self, model_path: str = os.path.join(os.path.dirname(__file__), "..", "models", "final_model_yolo11.pt")):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ Le modèle YOLOv11 n’a pas été trouvé à l’emplacement : {model_path}\n"
                f"📌 Veuillez vous assurer que le fichier .pt est bien présent."
            )
        try:
            self.model = YOLO(model_path)
            self.classes = self.model.names
            print(f"✅ Modèle chargé avec succès depuis : {model_path}")
        except Exception as e:
            raise RuntimeError(f"❌ Échec du chargement du modèle : {str(e)}")

    def process_image(self, image_np: np.ndarray, output_path: str = None) -> List[Dict]:
        """
        Traite une image NumPy, retourne les détections et peut enregistrer une version annotée.
        """
        results = self.model(image_np, conf=0.5)
        detections = []
        annotated_image = image_np.copy()

        for r in results:
            for box in r.boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.classes[class_id]

                    detections.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })

                    # Dessiner la boîte sur l’image
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {confidence:.2f}"
                    text_y = max(y1 - 10, 20)
                    cv2.putText(annotated_image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                except Exception as e:
                    print(f"⚠️ Erreur lors du traitement d'une boîte : {str(e)}")
                    continue

        if output_path:
            try:
                cv2.imwrite(output_path, annotated_image)
                print(f"🖼️ Image annotée sauvegardée dans {output_path}")
            except Exception as e:
                print(f"⚠️ Échec de sauvegarde d’image : {str(e)}")

        return detections

    def process_video(self, video_path: str, output_path: str) -> List[Dict]:
        """
        Traite une vidéo frame par frame et sauvegarde une vidéo annotée.
        """
        print(f"📹 Début du traitement vidéo : {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"❌ Impossible d'ouvrir la vidéo : {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # fallback si FPS == 0

        print(f"ℹ️ Dimensions : {width}x{height} @ {fps:.2f} FPS")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise RuntimeError(f"❌ Échec d'ouverture de VideoWriter pour {output_path}")

        all_detections = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.process_image(frame)
            all_detections.extend(detections)

            # Annoter la frame avec les détections
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                text_y = max(y1 - 10, 20)
                cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            out.write(frame)
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"🔁 {frame_count} frames traitées...")

        cap.release()
        out.release()

        print(f"✅ Vidéo annotée sauvegardée : {output_path} ({frame_count} frames)")
        return all_detections
