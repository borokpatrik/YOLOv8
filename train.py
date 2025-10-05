from ultralytics import YOLO
from datetime import datetime
import os

def main():
    # Create a unique folder name using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"yolov8n_localization_{timestamp}"

    # Load model
    model = YOLO('yolov8n.pt')

    # Train model
    model.train(
        data='localization.yaml',
        epochs=30,                          # Total training epochs
        imgsz=416,                          # Input image size
        batch=4,                            # Batch size
        project='E:/Coding/ML/Localizations/reference-YOLOv8n/result',  # Root folder for results
        name=run_name,                      # Unique subfolder name for this run
        exist_ok=True,                      # Allow overwriting if folder exists
        patience=15,                        # Early stopping if no improvement
        lr0=0.001,                          # Lower initial learning rate
        weight_decay=0.0005,                # Regularization to reduce overfitting
        augment=True,                       # Enable data augmentation
        hsv_h=0.015,                        # Color jittering (hue)
        hsv_s=0.7,                          # Color jittering (saturation)
        hsv_v=0.4,                          # Color jittering (value)
        flipud=0.1,                         # Vertical flip probability
        fliplr=0.5,                         # Horizontal flip probability
        mosaic=1.0,                         # Mosaic augmentation
        mixup=0.2,                          # Mixup augmentation
        dropout=0.1                         # Dropout in head layers
        # workers=0                         # Uncomment if needed for Windows compatibility
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
