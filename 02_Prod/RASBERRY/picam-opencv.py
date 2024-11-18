import cv2
import numpy as np
from picamera2 import Picamera2
from PIL import Image
from ultralytics import YOLO

# Charger le modèle YOLO (avec Edge TPU si applicable)
model = YOLO("judd_edgetpu.tflite")

# Initialiser la caméra Picamera2
picam2 = Picamera2()
picam2.start()

while True:
    # Capture une image en couleur (format BGR)
    frame = picam2.capture_array()

    # Convertir l'image de BGR à RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convertir l'image en PIL pour qu'elle soit compatible avec YOLO
    pil_image = Image.fromarray(frame_rgb)

    # Redimensionner l'image à la taille attendue par le modèle (ex: 640x640)
    pil_image = pil_image.resize((640, 640))

    # Exécuter la détection
    results = model(pil_image)

    # Afficher les résultats
    for result in results:
        result.show()

    # Vous pouvez aussi enregistrer les résultats si nécessaire
    results.save(filename="result.jpg")
