from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("rothko.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'