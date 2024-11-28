from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("rothko.pt")

# Export the model to TFLite format
model.export(format="ncnn")  # creates 'yolo11n_float32.tflite'


