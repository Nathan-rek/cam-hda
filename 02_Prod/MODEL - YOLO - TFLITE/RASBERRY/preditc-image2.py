import cv2

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("./runs/detect/train/weights/best.pt")

# Read an image using OpenCV
source = cv2.imread('rothkoanalyze.jpg')

# Run inference on the source
results = model(source)  # list of Results objects

# View results
for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes