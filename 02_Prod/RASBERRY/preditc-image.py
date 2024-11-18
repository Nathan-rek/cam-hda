from ultralytics import YOLO
from PIL import Image

# Load a pretrained YOLO11n model
model = YOLO("./runs/detect/train/weights/best.pt")

# Run inference on 'bus.jpg'
results = model(["orange-red-yellow-mark-rothko-artiste.jpg"], line_width=20)  # results list

for result in results:
    print(result.probs)
  
    
    result.show()
    result.save(filename="result.jpg")
        
    