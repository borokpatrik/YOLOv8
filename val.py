from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='localization.yaml',
    epochs=1,
    imgsz=416,
    batch=1,
    workers=0,
    plots=True
)
