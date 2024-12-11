import os
from ultralytics import YOLO
import cv2
import traceback
from picamera2 import Picamera2
import time

# Création d'un dossier pour sauvegarder les captures
capture_dir = 'capture'
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)
    print(f"Dossier {capture_dir} créé pour stocker les captures")

# Initialisation de la capture avec Picamera2
try:
    cap = Picamera2()
    cap.start()
    print("Capture de la caméra réussie")
except Exception as e:
    print(f"Erreur d'ouverture de la caméra : {e}")
    exit()

# Paramètres de capture
frame_width = cap.sensor_resolution[0]
frame_height = cap.sensor_resolution[1]
fps = 13

# Chargement du compteur de frames depuis le fichier
frame_count_file = "frame-count.txt"
if os.path.exists(frame_count_file):
    with open(frame_count_file, "r") as fichier:
        lastframe = int(fichier.read().strip())
else:
    lastframe = 0

frame_count = lastframe
print(f"Frame count initialisé à : {frame_count}")

# Initialisation de l'enregistrement vidéo
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f"output_{frame_count}.avi", fourcc, 6, (frame_width, frame_height))
print(f"Résolution webcam : {frame_width}x{frame_height}")
print(f"FPS webcam : {fps}")

# Chargement du modèle YOLO
model_path = "rothko.pt"
# model_path = input("Entrez le chemin du modèle YOLO : ")

assert os.path.exists(model_path), f"Fichier modèle non trouvé : {model_path}"
try:
    model = YOLO(model_path)
    print("Modèle chargé avec succès")
except Exception as e:
    print(f"Erreur de chargement du modèle : {str(e)}")
    print(f"Traceback complet : {traceback.format_exc()}")
    exit()

# Paramètre de confiance pour les détections
threshold = 0.1
# threshold = float(input("Entrez le seuil de confiance : "))

# FPS et timer
fps_counter = 0
start_time = time.time()

try:
    while True:
        try:
            # Capture de l'image en direct
            image = cap.capture_array()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Détection d'objets
            results = model(image)[0]
            detection_made = False

            # Dessin des boîtes englobantes et étiquettes sur l'image
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > threshold:
                    color = (0, 0, 0)  # Couleur verte
                    thickness = 1
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    label = f"{results.names[int(class_id)]}: {score:.2%}"
                    cv2.putText(image, label, (int(x1), int(y1 + 15)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, color, thickness)
                    detection_made = True
                    print(x1, x2, y1, y2)

            # Sauvegarder le cadre si une détection a été effectuée
            if detection_made:
                capture_filename = os.path.join(capture_dir, f"capture_{frame_count}.jpg")
                cv2.imwrite(capture_filename, image)
                print(f"Image sauvegardée : {capture_filename}")

            # Enregistrer le cadre dans la vidéo
            out.write(image)

            # Compteur FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - start_time >= 1:
                print(f"FPS : {fps_counter}")
                fps_counter = 0
                start_time = current_time

            # Incrémentation du compteur de frames
            frame_count += 1

            # Vérification du fichier stop.txt
            if os.path.exists("stop.txt"):
                with open("stop.txt", "r") as f:
                    if "stop" in f.read():
                        print("Arrêt détecté, arrêt du script.")
                        break

        except Exception as e:
            print(f"Erreur de traitement du cadre : {str(e)}")
            print(f"Traceback complet : {traceback.format_exc()}")

finally:
    # Sauvegarde du dernier frame_count dans le fichier
    with open(frame_count_file, "w") as fichier:
        fichier.write(str(frame_count))
    print(f"Frame count sauvegardé : {frame_count}")

    # Libérer les ressources
    cap.stop()
    out.release()
    print(" script terminé.")

