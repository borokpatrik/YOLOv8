from ultralytics import YOLO
import os
from datetime import datetime

def main():
    # Load your trained model
    model = YOLO('E:/Coding/ML/Localizations/reference-YOLOv8n/result/yolov8n_localization/weights/best.pt')

    # Create a unique folder name using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"yolov8n_test_{timestamp}"

    # Run inference on your test images
    results = model.predict(
        source='E:/Coding/ML/Localizations/dataset/test',  # Folder of test images
        imgsz=416,                                          # Match training size
        conf=0.25,                                          # Confidence threshold
        save=True,                                          # Save annotated images
        save_txt=True,                                      # Save predictions in YOLO format
        project='E:/Coding/ML/Localizations/reference-YOLOv8n/result',
        name=run_name,
        exist_ok=True
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
