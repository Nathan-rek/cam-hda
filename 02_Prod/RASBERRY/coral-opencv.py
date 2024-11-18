import cv2
import numpy as np
from picamera2 import Picamera2
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

# Paramètres
MODEL_PATH = "judd_edgetpu.tflite"  # Remplacez par le chemin de votre modèle EdgeTPU
CONFIDENCE_THRESHOLD = 0.5  # Seuil de confiance pour la détection

# Initialisation de la caméra
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Charger le modèle EdgeTPU
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
print("Modèle EdgeTPU chargé avec succès")

# Boucle de détection en temps réel
try:
    while True:
        # Capture une image depuis la Picamera2
        image = picam2.capture_array()

        # Prétraitement de l'image pour l'inférence (redimensionnement et conversion en RGB)
        input_shape = common.input_size(interpreter)
        image_resized = cv2.resize(image, input_shape)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Placer l'image dans le tenseur d'entrée
        common.set_input(interpreter, image_rgb)
        
        # Lancer l'inférence
        interpreter.invoke()

        # Récupérer les résultats de détection
        objects = detect.get_objects(interpreter, CONFIDENCE_THRESHOLD)
        
        # Afficher les résultats de détection sur l'image
        for obj in objects:
            bbox = obj.bbox
            label = f"ID {obj.id}, Score {obj.score:.2f}"
            
            # Dessiner la boîte de détection et le label
            cv2.rectangle(image, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
            cv2.putText(image, label, (bbox.xmin, bbox.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Afficher l'image avec détection
        cv2.imshow("Détection en temps réel", image)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Erreur durant l'inférence ou l'affichage : {e}")

finally:
    # Libérer les ressources
    picam2.stop()
    cv2.destroyAllWindows()
