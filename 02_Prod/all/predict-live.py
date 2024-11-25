import os
from ultralytics import YOLO
import cv2
import traceback
import time

# Création d'un dossier pour sauvegarder les captures
capture_dir = 'capture'
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)
    print(f"Dossier {capture_dir} créé pour stocker les captures")

# Initialisation de la capture webcam
try:
    cap = cv2.VideoCapture(1)
    print("Capture webcam réussie")
except Exception as e:
    print(f"Erreur d'ouverture de la webcam : {e}")
    exit()
    


# Récupération des dimensions du flux vidéo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


fichier = open("frame-count.txt", "r")
   
lastframe = fichier.read().strip()

lastframe = int(lastframe)


    
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f"output_{lastframe}.avi", fourcc, 6, (640,  480))

print(f"Résolution webcam : {frame_width}x{frame_height}")
print(f"FPS webcam : {fps}")



# Chargement du modèle

model = input()
model_path = os.path.join(model)
print(f"Chemin du modèle : {model_path}")
assert os.path.exists(model_path), f"Fichier modèle non trouvé : {model_path}"

try:
    model = YOLO(model_path)  # chargement du modèle personnalisé
    print("Modèle chargé avec succès")
except Exception as e:
    print(f"Erreur de chargement du modèle : {str(e)}")
    print(f"Traceback complet : {traceback.format_exc()}")
    exit()
    

# Paramètres
threshold = float(input())
frame_count = 0 + lastframe
fps_counter = 0
start_time = time.time()


while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            print("Échec de la lecture du cadre")
            break
        
        # Détection d'objets
        results = model(frame)[0]

        detection_made = False

        # Dessin des boîtes englobantes et étiquettes sur le cadre
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                color = (0, 0, 0)  # Couleur verte pour les boîtes englobantes
                thickness = 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                confidence_percentage = f"{score * 100:.2g}%"
#                 confidence_percentage = int(confidence_percentage)
                label = f"{results.names[int(class_id)]}: {confidence_percentage}"
                
                if y1 or y2 < frame_height:
                    cv2.putText(frame, label, (int(x1), int(y1 - 20)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1, cv2.LINE_AA)
                
                else:
                    cv2.putText(frame, label, (int(x1), int(y1 + 15)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1, cv2.LINE_AA)
                    
        
                detection_made = True  # Indiquer qu'une détection a été faite
                print(x1, x2, y1, y2)
                
#                 if x1 or x2 > frame_width:
#                     cv2.putText(frame, label, (int(x1 + 15), int(y1 + 15)),
#                         cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1, cv2.LINE_AA)
#                 if y1 or y2 > frame_height:
#                     cv2.putText(frame, label, (int(x1 + 15), int(y1 + 15)),
#                         cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1, cv2.LINE_AA)
# 
                    
                    

        # Sauvegarder le cadre si une détection a été effectuée
        if detection_made:
            capture_filename = os.path.join(capture_dir, f"capture_{frame_count}.jpg")
            cv2.imwrite(capture_filename, frame)
            print(f"Image sauvegardée : {capture_filename}")
            

        # Affichage du cadre résultant
        cv2.imshow('Détection en temps réel', frame)

        # Compteur FPS
        fps_counter += 1
        current_time = time.time()
        
        if current_time - start_time >= 1:
            print(f"FPS : {fps_counter}")
            fps_counter = 0
            start_time = current_time

        # Informations sur la détection
        if frame_count % 10 == 0:
            num_detections = len(results.boxes.data.tolist())
            avg_inference_time = sum(results.speed.values()) / len(results.speed)
            print(f"{frame_count} : {num_detections} détections, Temps d'inférence moyen : {avg_inference_time:.1f}ms")

        # Sortie sur pression de 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            fichier = open("frame-count.txt", "r+")
            fichier.seek(0)
            fichier.write(str(frame_count))
            fichier.close()
            break
        
        frame_count += 1
        
        
        print(frame_count)
        

    except Exception as e:
        print(f"Erreur de traitement du cadre : {str(e)}")
        print(f"Traceback complet : {traceback.format_exc()}")
    out.write(frame) 
    
# Libérer la webcam et fermer les fenêtres
cap.release()
out.release()
cv2.destroyAllWindows()
