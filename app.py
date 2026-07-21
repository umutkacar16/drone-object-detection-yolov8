import sys
import os
import cv2
import torch
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QFileDialog, QFrame)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO

class DroneDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # Dinamik model arama
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(BASE_DIR, "runs", "detect", "drone_yolov8_s_run", "weights", "best.pt")
        
        # Yedek varsayılan model yolu
        if not os.path.exists(self.model_path):
            self.model_path = "best.pt"
            
        self.dataset_name = "merged_dataset (VisDrone + Animals)"
        
        print(f"-> Model yükleniyor: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Drone Tabanlı Canlı Nesne Tespit Sistemi")
        self.setGeometry(100, 100, 1300, 750)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI';")
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # SOL PANEL
        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)
        
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: #252538; border-radius: 12px; padding: 18px;")
        info_layout = QVBoxLayout(info_frame)
        
        title_label = QLabel("📊 SİSTEM PARAMETRELERİ")
        title_label.setStyleSheet("font-weight: bold; font-size: 18px; color: #89b4fa;")
        info_layout.addWidget(title_label)
        
        model_label = QLabel("<b>Kullanılan Model:</b><br><font color='#a6e3a1'>YOLOv8s (Small Mimarisi)</font>")
        info_layout.addWidget(model_label)
        
        dataset_label = QLabel(f"<b>Çalışılan Veri Seti:</b><br><font color='#f9e2af'>{self.dataset_name}</font>")
        info_layout.addWidget(dataset_label)
        
        classes_label = QLabel("<b>Hedef Sınıflar:</b><br>• İnsan (%94.0)<br>• Kuş (%89.3)<br>• Evcil Hayvan (%94.9)<br>• Vahşi Hayvan (%97.8)")
        info_layout.addWidget(classes_label)
        
        left_panel.addWidget(info_frame)
        
        perf_frame = QFrame()
        perf_frame.setStyleSheet("background-color: #252538; border-radius: 12px; padding: 18px;")
        perf_layout = QVBoxLayout(perf_frame)
        
        perf_title = QLabel("⚡ DONANIM & PERFORMANS")
        perf_title.setStyleSheet("font-weight: bold; font-size: 18px; color: #f9e2af;")
        perf_layout.addWidget(perf_title)
        
        cuda_status = "CUDA Aktif 🚀" if torch.cuda.is_available() else "CPU Modu ⚠️"
        gpu_label = QLabel(f"<b>Donanım:</b> RTX 4070 Laptop GPU<br><font color='#a6e3a1'><b>Durum:</b> {cuda_status}</font>")
        perf_layout.addWidget(gpu_label)
        
        map_label = QLabel("<b>Genel Doğruluk (mAP50):</b><br><font color='#89b4fa' size='5'><b>%94.0</b></font>")
        perf_layout.addWidget(map_label)
        
        speed_label = QLabel("<b>Çıkarım Hızı:</b><br><font color='#f38ba8' size='5'><b>2.9 ms</b></font> <font color='#a6adc8' size='3'>(~340 FPS Teorik)</font>")
        perf_layout.addWidget(speed_label)
        
        left_panel.addWidget(perf_frame)
        
        self.btn_open = QPushButton("📸 Resim / Video Seç")
        self.btn_open.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa; color: #11111b; font-weight: bold; font-size: 16px;
                border-radius: 10px; padding: 16px;
            }
            QPushButton:hover { background-color: #b4befe; }
        """)
        self.btn_open.clicked.connect(self.open_file)
        left_panel.addWidget(self.btn_open)
        
        # SAĞ PANEL
        right_panel = QVBoxLayout()
        self.video_label = QLabel("Lütfen test etmek için bir görsel veya video dosyası seçin.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            border: 2px dashed #45475a; border-radius: 12px; 
            background-color: #11111b; font-size: 16px; color: #6c7086;
        """)
        self.video_label.setMinimumSize(800, 600)
        right_panel.addWidget(self.video_label)
        
        main_layout.addLayout(left_panel, 32)
        main_layout.addLayout(right_panel, 68)
        self.setLayout(main_layout)

    def open_file(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Dosya Seç", "", "Medya Dosyaları (*.jpg *.jpeg *.png *.mp4 *.avi)"
        )
        if file_path:
            if file_path.endswith(('.mp4', '.avi')):
                self.start_video_stream(file_path)
            else:
                self.process_image(file_path)

    def process_image(self, img_path):
        results = self.model(img_path)
        annotated_frame = results[0].plot()
        self.display_frame(annotated_frame)

    def start_video_stream(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.timer.start(30) # ~33 FPS canlı işleme

    def update_video_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                results = self.model(frame)
                annotated_frame = results[0].plot()
                self.display_frame(annotated_frame)
            else:
                self.timer.stop()
                self.cap.release()

    def display_frame(self, frame_bgr):
        rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DroneDetectionApp()
    ex.show()
    sys.exit(app.exec_())
    