import os
import cv2
import time
import traceback
import numpy as np
from picamera2 import Picamera2
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

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

# Charger le modèle TensorFlow Lite optimisé pour Edge TPU
model_path = "judd_edgetpu.tflite"  # Remplacez par le chemin vers votre modèle TensorFlow Lite compilé pour Edge TPU
assert os.path.exists(model_path), f"Fichier modèle non trouvé : {model_path}"

try:
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    print("Modèle chargé avec succès")
except Exception as e:
    print(f"Erreur de chargement du modèle : {str(e)}")
    print(f"Traceback complet : {traceback.format_exc()}")
    exit()

# Fonction pour effectuer des inférences avec le modèle Edge TPU
def run_inference(image):
    # Prétraitement de l'image pour l'adapter au modèle TensorFlow Lite
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (300, 300))  # Exemple pour un modèle de 300x300
    image_resized = np.expand_dims(image_resized, axis=0).astype(np.float32)

    # Alimentation du modèle et récupération des résultats
    common.set_input(interpreter, image_resized)
    interpreter.invoke()
    output_data = common.output_tensor(interpreter, 0)

    return output_data

# Paramètre de confiance pour les détections
threshold = 0.1

# FPS et timer
fps_counter = 0
start_time = time.time()

try:
    while True:
        try:
            # Capture de l'image en direct
            image = cap.capture_array()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Détection d'objets avec le modèle Edge TPU
            results = run_inference(image)
            detection_made = False

            # Dessin des boîtes englobantes et étiquettes sur l'image
            for result in results:
                x1, y1, x2, y2, score, class_id = result
                if score > threshold:
                    color = (0, 0, 0)  # Couleur verte
                    thickness = 1
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    label = f"ID {int(class_id)}: {score:.2%}"  # Utilisez l'ID de la classe ou le nom de la classe
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
    print("script terminé.")

