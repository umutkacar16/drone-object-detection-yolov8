import os
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(BASE_DIR, "data.yaml")

MODEL_TYPE = "yolov8" 
MODEL_SIZE = "s"      
MODEL_NAME = f"{MODEL_TYPE}{MODEL_SIZE}.pt"

def start_training():
    print(f"-> {MODEL_NAME} mimarisi yükleniyor...")
    model = YOLO(MODEL_NAME)
    
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"-> Eğitim Cihazı: {device}")
    
    results = model.train(
        data=YAML_PATH,
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        workers=4,
        name=f"drone_{MODEL_TYPE}_{MODEL_SIZE}_run",
        save=True,
        plots=True
    )
    
    print("\n[TAMAMLANDI] Model eğitimi başarıyla sonuçlandı!")

if __name__ == "__main__":
    start_training()
    