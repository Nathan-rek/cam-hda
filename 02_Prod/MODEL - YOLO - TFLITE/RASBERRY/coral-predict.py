import os
import cv2
import numpy as np
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import image_processing as img_utils
import time

# Chemins vers les fichiers
VIDEO_PATH = '/home/pi/Documents/nathan/cam-hda/videos/test.mp4'
MODEL_PATH = '/home/pi/Documents/nathan/cam-hda/models/judd_edgetpu.tflite'
LABELS_PATH = '/home/pi/Documents/nathan/cam-hda/models/judd_labels.txt'

# Création du moteur de détection
engine = DetectionEngine(MODEL_PATH)
labels = {}
with open(LABELS_PATH, 'r') as f:
    pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
    labels = {int(k): v for k, v in pairs}

# Ouvre la vidéo
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Impossible d'ouvrir le fichier vidéo")
    exit()

# Paramètres de la vidéo
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Création du writer pour la sortie vidéo
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

frame_count = 0
start_time = time.time()

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            break

        # Prétraitement de l'image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        padded_rgb_frame = img_utils.pad_to_square(rgb_frame)
        scaled_frame = img_utils.resize_image(padded_rgb_frame, engine.get_input_tensor_shape()[1:3])

        # Exécution de l'inference
        start_infer_time = time.time()
        _, raw_result = engine.run_inference(scaled_frame.tobytes())
        end_infer_time = time.time()

        # Traitement des résultats
        for score, label_index, coord in raw_result:
            if score > 0.5:  # Seuil de confiance
                x0, y0, x1, y1 = coord
                x, y, w, h = img_utils.bounding_box_coordinates(
                    x0, y0, x1, y1, padded_rgb_frame.shape[:2]
                )
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                cv2.putText(frame, labels[label_index], (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Affichage et sauvegarde du frame
        cv2.imshow('Détection en temps réel', frame)
        out.write(frame)

        # Calcul du FPS
        frame_count += 1
        current_time = time.time()
        if current_time - start_time >= 1:
            fps = frame_count
            frame_count = 0
            start_time = current_time
        cv2.putText(frame, f"FPS: {fps}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Sortie sur pression de 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Erreur lors du traitement du frame : {str(e)}")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Traitement terminé.")
