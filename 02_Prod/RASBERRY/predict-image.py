


from ultralytics import YOLO
from PIL import Image

# Load YOLO model and specify task (if needed)
model = YOLO("judd_edgetpu.tflite", task="detect")

# Run inference on 'judd-img.jpg'
results = model(["judd-img.jpg"], line_width=20)  # inference on the image

for result in results:
    # Access detection results - boxes, classes, scores, etc.
    if result.boxes is not None:
        for box in result.boxes:
            print(f"Class: {box.cls}, Confidence: {box.conf}, Box coordinates: {box.xyxy}")

    # Save the result image with annotations
    result.save("result.jpg")
