# 🛸 Drone-Based Object Detection System (YOLOv8s & PyQt5)

A high-performance, real-time small object detection system designed for drone imagery using **YOLOv8s** architecture and an interactive **PyQt5** graphical user interface.

## 📊 Key Performance Metrics
* **mAP50:** 84.0%
* **Inference Speed:** 2.9 ms (~250+ FPS)
* **Target Classes:** Human, Bird, Animal

## 📁 Repository Structure
* `app.py`: PyQt5 desktop interface for image/video testing.
* `dataset.py`: Dataset preparation and label mapping pipeline.
* `model.py`: Model training script with YOLOv8s configuration.
* `data.yaml`: Dataset split paths and class definitions.
* `assets/`: Contains training plots, evaluation metrics, and documentation assets.

## 💾 Large Assets & Dataset Access
Due to file size limitations, trained model weights, full runs data, and the merged dataset are hosted on Google Drive:

* 📦 **Merged Dataset:** [Download from Google Drive](https://drive.google.com/drive/folders/19s28dprd1idCTuMIlx7coB78bazHZwEF?usp=sharing)
* 🎯 **Trained Weights & Training Runs (`runs/` & `best.pt`):** [Download from Google Drive](https://drive.google.com/drive/folders/1wp_3qtlECz9Wax3X-6jdlMyVQT1i08Bf?usp=sharing)

## 📊 Training Results 
![Training Results](assets/100%20epoch/results.png)

## 🚀 Quick Start

### 1. Installation
```bash
git clone [https://github.com/umutkacar16/drone-object-detection-yolov8.git](https://github.com/umutkacar16/drone-object-detection-yolov8.git)
cd drone-object-detection-yolov8
pip install -r requirements.txt
