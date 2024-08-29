from ultralytics import YOLO
import torch

torch.cuda.empty_cache()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    model = YOLO('yolov8n.pt')
    
    # Training.
    results = model.train(
       data='./config.yaml',
       task='detect',
       imgsz=640,
       epochs=50,
       batch=4
    )