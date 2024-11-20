import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("rothko.pt")

# Export the model to TFLite format using TensorFlow directly
import tensorflow as tf

converter = tf.lite.TFLiteConverter(v2:0)
tflite_model = converter.convert(model.model)

with open('yolo11n_float32.tflite', 'wb') as f:
    f.write(tflite_model)

# Load the exported TFLite model
tflite_model = YOLO("yolo11n_float32.tflite")

# Run inference
results = tflite_model("https://ultralytics.com/images/bus.jpg")
