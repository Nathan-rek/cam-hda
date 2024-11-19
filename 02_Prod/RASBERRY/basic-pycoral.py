from ultralytics import YOLO

# Load a model
model = YOLO("judd_edgetpu.tflite")  # Load an official model or custom model

# Run Prediction
model.predict("judd-img.jpg")
