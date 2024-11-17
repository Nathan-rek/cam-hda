import os
from ultralytics import YOLO
import cv2
import traceback

# Create videos directory if it doesn't exist
VIDEOS_DIR = os.path.join('.', 'videos')
os.makedirs(VIDEOS_DIR, exist_ok=True)
print(f"Videos directory created: {VIDEOS_DIR}")

# Set correct paths
video_path = os.path.join(VIDEOS_DIR, "C:/Users/prost/OneDrive/Documents/SCHOOL/ERG/Internet-et-programations/B3/camera-HDA/YOLOv8/rohtko2/02_Prod/render-song+img.mp4")
print(f"Video path: {video_path}")
assert os.path.exists(video_path), f"Video file not found: {video_path}"

out_path = '{}_out.mp4'.format(video_path)
print(f"Output path: {out_path}")

# Find the correct model path
model_path = os.path.join('runs', 'detect', 'train', 'weights', 'best.pt')
print(f"Model path: {model_path}")
assert os.path.exists(model_path), f"Model file not found: {model_path}"

try:
    cap = cv2.VideoCapture(video_path)
    print("Video capture successful")
except Exception as e:
    print(f"Error opening video: {e}")
    exit()

ret, frame = cap.read()
if not ret:
    print("Failed to read frame")
else:
    print(f"Frame shape: {frame.shape}")

# Load a model
try:
    model = YOLO(model_path)  # load a custom model
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print(f"Full traceback: {traceback.format_exc()}")
    exit()

threshold = 0.5

out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
print(f"Output video initialized at {out_path}")

frame_count = 0

while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
        
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                color = (0, 0, 0)
                thickness = 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                confidence_percentage = f"{score * 100:.2f}%"
                label = f"{results.names[int(class_id)]}: {confidence_percentage}"
                cv2.putText(frame, label, (int(x1), int(y1 - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2, cv2.LINE_AA)

        out.write(frame)
        
        # Print speed information every 10 frames
        if frame_count % 10 == 0:
            print(f"{frame_count}: {results.shapes[0]}x{results.shapes[1]} ({len(results.boxes)}) detections, {results.times.avg:.1f}ms")

    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
    
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
