---
title: API Détection Postures Chiens
emoji: 🐶
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
# 🐶 API de Détection de Postures de Chiens avec YOLOv11

Une API RESTful basée sur **FastAPI** pour détecter les **postures de chiens** (assis, debout, couché, etc.) dans des images, vidéos ou flux webcam, à l’aide d’un modèle **YOLOv11** pré-entraîné.

---

## 📚 Table des matières
- Fonctionnalités
- Structure du Projet
- Prérequis
- Installation
- Utilisation
- Endpoints de l'API
- Fonctionnement Interne

---

## ✅ Fonctionnalités

- 🔍 **Détection sur image** : Téléversez une image, obtenez une image annotée, un JSON ou un CSV.
- 🎥 **Détection sur vidéo** : Téléversez une vidéo, recevez une version annotée.
- 📡 **Streaming en temps réel (Webcam)** : Visualisez les détections en direct via WebSocket.
- 🧠 **Modèle intégré** : Utilise un modèle YOLOv11 (`final_model_yolo11.pt`) optimisé pour la posture canine.

---

## 📁 Structure du Projet

```
.
├── Dockerfile
├── README.md
├── requirements.txt
├── api/
│   ├── main.py
│   ├── detectors/
│   │   └── detectors_yolo11.py
│   ├── models/
│   │   └── final_model_yolo11.pt
│   ├── routers/
│   │   └── routers_yolo11.py
│   ├── schemas/
│   │   └── schemas_yolo11.py
│   └── templates/
│       └── index.html
├── uploads/
├── temp_results/
└── .gitignore / .dockerignore / .gitattributes
```

---

## ⚙️ Prérequis

- Python ≥ 3.8
- Pip
- Modules : `fastapi`, `uvicorn`, `opencv-python`, `ultralytics`, etc.

---

## 🚀 Installation

1. **Clonez le dépôt** :
   ```bash
   git clone https://github.com/<votre-utilisateur>/<nom-du-depot>.git
   cd <nom-du-depot>
   ```

2. **Créez un environnement virtuel** :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```

3. **Installez les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

4. **Assurez-vous que le modèle YOLOv11 est disponible** :
   Le fichier `final_model_yolo11.pt` doit être présent dans `api/models/`.

---

## ▶️ Utilisation

```bash
cd api
uvicorn api.main:app --reload
```

Accédez à :
- URL : http://127.0.0.1:8000/docs (ctrl+c)
---

## 🧪 Endpoints de l’API

### 📸 Détection sur image

- `POST /yolo/predict` avec `file` (image) + `output_format` (image, json, csv)

### 🎬 Détection sur vidéo

- `POST /yolo/predict-video` avec `file` (vidéo)

### 🔴 Streaming Webcam

- `POST /yolo/start-stream` / `POST /yolo/stop-stream`
- WebSocket : `WS /yolo/ws`
- Exemple Web : http://127.0.0.1:8000/

---

## 🧠 Fonctionnement Interne

- `main.py` : démarre l’API et charge le modèle
- `detectors_yolo11.py` : logique de détection
- `routers_yolo11.py` : endpoints
- `schemas_yolo11.py` : validation Pydantic

---

## ✅ Conclusion

Ce projet démontre la puissance de l’intégration entre **FastAPI** et **YOLOv11** pour la détection intelligente de postures canines à partir d’images, de vidéos ou de flux en temps réel. 