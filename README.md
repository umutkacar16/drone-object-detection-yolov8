# 🛸 Drone-Based Object Detection System (YOLOv8s & PyQt5)

A high-performance, real-time small object detection system designed for drone imagery using **YOLOv8s** architecture and an interactive **PyQt5** graphical user interface.

## 📊 Key Performance Metrics
* **mAP50:** 94.0%
* **Inference Speed:** 2.9 ms (~250+ FPS)
* **Target Classes:** Human, Bird, Pet, Wild Animal

## 📁 Repository Structure
* `app.py`: PyQt5 desktop interface for image/video testing.
* `dataset.py`: Dataset preparation and label mapping pipeline.
* `model.py`: Model training script with YOLOv8s configuration.
* `data.yaml`: Dataset split paths and class definitions.

## 📊 Training Results 
![Training Results](runs\detect\drone_yolov8_s_run/results.png)



## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/umutkacar16/drone-object-detection-yolov8.git
cd drone-object-detection-yolov8
pip install -r requirements.txt





