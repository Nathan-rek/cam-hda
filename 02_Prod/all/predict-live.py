import os
from ultralytics import YOLO
import cv2
import traceback
import time
import socket
import pygame
import threading
import wave
import pyaudio


# Création d'un dossier pour sauvegarder les captures
capture_dir = 'capture'
os.makedirs(capture_dir, exist_ok=True)

# Initialisation de la capture webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Erreur d'ouverture de la webcam")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Chargement du compteur lastframe
lastframe_file = "frame-count.txt"
if not os.path.exists(lastframe_file):
    with open(lastframe_file, "w") as f:
        f.write("0")
with open(lastframe_file, "r") as f:
    lastframe = int(f.read().strip())

# Configuration de la sortie vidéo
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_filename = f"output_{lastframe}.avi"
out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

print(f"Vidéo enregistrée sous : {video_filename}")

# Chargement du modèle YOLO
model_path = input("Nom du modèle YOLO : ")
assert os.path.exists(model_path), f"Fichier modèle non trouvé : {model_path}"
model = YOLO(model_path)

# Paramètres
threshold = float(input("Seuil de confiance : "))
frame_count = lastframe

# Configuration de pygame pour la musique
pygame.init()
pygame.mixer.init()

music_file = "music.mp3"
assert os.path.exists(music_file), f"Fichier musique non trouvé : {music_file}"
pygame.mixer.music.load(music_file)
pygame.mixer.music.play(-1)

# Configuration de PyAudio pour capturer l'audio
audio = pyaudio.PyAudio()
audio_filename = f"audio_{lastframe}.wav"

# Paramètres audio
audio_format = pyaudio.paInt16
channels = 2
rate = 44100
chunk = 1024

# Fonction pour capturer l'audio
def record_audio():
    with wave.open(audio_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(audio_format))
        wf.setframerate(rate)

        stream = audio.open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
        print("Enregistrement audio commencé...")
        try:
            while running:
                data = stream.read(chunk)
                wf.writeframes(data)
        finally:
            stream.stop_stream()
            stream.close()
            print("Enregistrement audio terminé.")

# Thread pour l'enregistrement audio
running = True
audio_thread = threading.Thread(target=record_audio)
audio_thread.start()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Détection d'objets
        results = model(frame)[0]
        max_score = 0.0

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{results.names[int(class_id)]}: {score:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                max_score = max(max_score, score)

        # Ajustement du volume
        volume = max_score
        pygame.mixer.music.set_volume(volume)
        print(f"Volume : {volume * 100:.1f}% (Score max : {max_score:.2f})")

        # Enregistrement du cadre
        out.write(frame)

        # Affichage
        cv2.imshow('Détection en temps réel', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

finally:
    running = False
    audio_thread.join()

    # Sauvegarde de lastframe
    with open(lastframe_file, "w") as f:
        f.write(str(frame_count))

    # Libération des ressources
    cap.release()
    out.release()
    pygame.mixer.music.stop()
    pygame.quit()
    audio.terminate()
    cv2.destroyAllWindows()

# Fusion vidéo et audio avec FFmpeg
output_filename = f"output_with_audio_{lastframe}.avi"
os.system(f"ffmpeg -i {video_filename} -i {audio_filename} -c:v copy -c:a aac {output_filename}")

print(f"Vidéo générée avec audio : {output_filename}")
