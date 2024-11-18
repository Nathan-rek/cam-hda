import cv2
from picamera2 import Picamera2
import time

# Initialisation de la caméra
try:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    print("Caméra initialisée et démarrée avec succès.")

    # Pause pour laisser le capteur se stabiliser
    time.sleep(2)

    # Capture d'une seule image
    image = picam2.capture_array()
    print("Image capturée avec succès.")

    # Enregistrer l'image
    cv2.imwrite("test_capture.jpg", image)
    print("Image enregistrée sous 'test_capture.jpg'.")

    # Charger et afficher l'image capturée
    image = cv2.imread("test_capture.jpg")
    cv2.imshow("Image Test", image)
    cv2.waitKey(0)  # Attente d'une touche
    cv2.destroyAllWindows()  # Fermeture des fenêtres d'affichage

except Exception as e:
    print(f"Erreur lors de l'initialisation ou de l'utilisation de la caméra : {e}")

finally:
    # Arrêter la caméra si elle a été initialisée
    if 'picam2' in locals():
        picam2.stop()
