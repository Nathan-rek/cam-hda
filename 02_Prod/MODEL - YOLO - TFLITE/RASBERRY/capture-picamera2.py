# capture_image.py (exécuté en dehors du venv)

from picamera2 import Picamera2
import time
import numpy as np
import cv2

def capture_image():
    # Initialisation de la caméra
    picam2 = Picamera2()
    picam2.start()

    while True:
        # Capture d'une image
        frame = picam2.capture_array()
        
        # Sauvegarder l'image dans un fichier temporaire
        timestamp = time.time()
        filename = f"/tmp/captured_image_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image sauvegardée : {filename}")

        # Attendez un moment avant de prendre la prochaine image
        time.sleep(1)

if __name__ == "__main__":
    capture_image()
