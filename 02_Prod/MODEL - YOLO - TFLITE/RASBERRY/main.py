from ultralytics import YOLO
import torch

# Vérifiez si un GPU est disponible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Chargez le modèle
model = YOLO("yolov8n.yaml")

# Entraînez le modèle en utilisant le GPU
model.train(data="config.yaml", epochs=20)

model.export(format="edgetpu")

print(epochs)
