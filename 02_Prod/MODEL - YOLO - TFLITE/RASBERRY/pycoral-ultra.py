import time
import cv2
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter

# Fonction pour dessiner les objets détectés
def draw_objects(draw, objs, labels):
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline='red')
        draw.text((bbox.xmin, bbox.ymin - 10), f'{labels.get(obj.id, obj.id)}: {obj.score:.2f}', fill='red')

# Fonction principale pour la détection en direct
def live_detection(model_path, threshold=0.4):
    # Initialiser la caméra avec PyCamera2 (sans libcamera)
    import picamera2
    picam2 = picamera2.Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    print("Démarrage de la détection en direct...")

    # Charger le modèle PyCoral
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    try:
        while True:
            # Capturer une image
            frame = picam2.capture_array()
            # Convertir en format PIL
            image = Image.fromarray(frame)

            # Redimensionner l'image pour l'inférence
            _, scale = common.set_resized_input(
                interpreter, image.size, lambda size: image.resize(size, Image.LANCZOS))

            # Exécuter l'inférence
            start = time.perf_counter()
            interpreter.invoke()
            inference_time = time.perf_counter() - start
            objs = detect.get_objects(interpreter, threshold, scale)
            print(f"Temps d'inférence : {inference_time * 1000:.2f} ms")

            # Dessiner les objets détectés
            image = image.convert('RGB')
            draw_objects(ImageDraw.Draw(image), objs, labels={})

            # Afficher l'image avec OpenCV
            cv2.imshow("Détection en temps réel", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

            # Sortir si la touche 'q' est pressée
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "judd_edgetpu.tflite"  # Remplacez par le chemin de votre modèle EdgeTPU
    live_detection(model_path, threshold=0.4)
