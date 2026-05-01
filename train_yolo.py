from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='aerial_yolo_model'
)

# Validate
results = model.val()

print("YOLOv8 training and validation complete.")
