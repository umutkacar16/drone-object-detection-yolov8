import sys
import os
import time
import cv2
import torch
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QFileDialog, QFrame, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO

class DroneDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(BASE_DIR, "runs", "detect", "drone_yolov8_s_run", "weights", "best.pt")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ MODEL BULUNAMADI! Aranan yol: {self.model_path}")
            
        self.dataset_name = "merged_dataset (VisDrone + Birds + WAID)"
        
        print(f"-> Model yükleniyor: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)
        self.prev_time = time.time()
        self.is_webcam = False
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Drone Tabanlı Canlı Nesne Tespit ve İzleme Sistemi")
        self.setGeometry(80, 80, 1400, 850)
        self.setStyleSheet("background-color: #181825; color: #cdd6f4; font-family: 'Segoe UI', Arial;")
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # ================= SOL PANEL =================
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        left_panel.setContentsMargins(0, 0, 0, 0)
        
        # 1. SİSTEM MİMARİSİ VE VERİ SETİ AÇIKLAMA KARTI
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: #1e1e2e; border-radius: 10px; padding: 12px;")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(4)
        
        title_label = QLabel("📊 SİSTEM MİMARİSİ & VERİ SETİ")
        title_label.setStyleSheet("font-weight: bold; font-size: 15px; color: #89b4fa;")
        info_layout.addWidget(title_label)
        
        info_layout.addWidget(QLabel("<b>Kullanılan Model:</b> <font color='#a6e3a1'>YOLOv8s (Small)</font>"))
        info_layout.addWidget(QLabel(f"<b>Veri Seti:</b> <font color='#f9e2af'>{self.dataset_name}</font>"))
        info_layout.addWidget(QLabel("<b>Giriş Çözünürlüğü:</b> <font color='#b4befe'>800 x 800 px</font>"))
        
        # Train & Val Bölünme Açıklamaları
        train_info = QLabel("<b>• Train (Eğitim):</b> <font color='#a6e3a1'>12,157 Görsel (%82.5)</font><br>"
                            "<font color='#9399b2'><i>(Modelin ağırlıklarını öğrendiği ana veri seti)</i></font>")
        train_info.setWordWrap(True)
        info_layout.addWidget(train_info)
        
        val_info = QLabel("<b>• Validate (Doğrulama):</b> <font color='#89b4fa'>2,589 Görsel (%17.5)</font><br>"
                          "<font color='#9399b2'><i>(Modelin doğruluğunun test edildiği bağımsız set)</i></font>")
        val_info.setWordWrap(True)
        info_layout.addWidget(val_info)
        
        left_panel.addWidget(info_frame, 3)
        
        # 2. ERKEN UYARI & SÜRÜ BİLDİRİMİ KARTI
        alert_frame = QFrame()
        alert_frame.setStyleSheet("background-color: #1e1e2e; border-radius: 10px; padding: 12px;")
        alert_layout = QVBoxLayout(alert_frame)
        alert_layout.setSpacing(4)
        
        alert_title = QLabel("🚨 ERKEN UYARI & SÜRÜ BİLDİRİMİ")
        alert_title.setStyleSheet("font-weight: bold; font-size: 15px; color: #f38ba8;")
        alert_layout.addWidget(alert_title)
        
        self.alert_label = QLabel("<b>Sürü Durumu:</b> <font color='#a6e3a1'>Normal (0 Kuş)</font>")
        self.alert_label.setStyleSheet("font-size: 14px;")
        alert_layout.addWidget(self.alert_label)
        
        left_panel.addWidget(alert_frame, 1)

        # 3. ANLIK TESPİT SAYILARI KARTI
        stats_frame = QFrame()
        stats_frame.setStyleSheet("background-color: #1e1e2e; border-radius: 10px; padding: 12px;")
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setSpacing(4)
        
        stats_title = QLabel("🎯 ANLIK TESPİT SAYILARI")
        stats_title.setStyleSheet("font-weight: bold; font-size: 15px; color: #a6e3a1;")
        stats_layout.addWidget(stats_title)
        
        self.lbl_human_count = QLabel("<b>• İnsan Tespiti:</b> <font color='#89b4fa'>0</font>")
        self.lbl_bird_count = QLabel("<b>• Kuş Tespiti:</b> <font color='#f9e2af'>0</font>")
        self.lbl_animal_count = QLabel("<b>• Hayvan Tespiti:</b> <font color='#a6e3a1'>0</font>")
        
        stats_layout.addWidget(self.lbl_human_count)
        stats_layout.addWidget(self.lbl_bird_count)
        stats_layout.addWidget(self.lbl_animal_count)
        
        left_panel.addWidget(stats_frame, 2)

        # 4. DONANIM & AKIŞ METRİKLERİ KARTI
        perf_frame = QFrame()
        perf_frame.setStyleSheet("background-color: #1e1e2e; border-radius: 10px; padding: 12px;")
        perf_layout = QVBoxLayout(perf_frame)
        perf_layout.setSpacing(4)
        
        perf_title = QLabel("⚡ DONANIM & AKIŞ METRİKLERİ")
        perf_title.setStyleSheet("font-weight: bold; font-size: 15px; color: #f9e2af;")
        perf_layout.addWidget(perf_title)
        
        cuda_status = "CUDA Aktif 🚀" if torch.cuda.is_available() else "CPU Modu ⚠️"
        gpu_label = QLabel(f"<b>Donanım:</b> RTX 4070 Laptop GPU<br><b>Hızlandırıcı:</b> <font color='#a6e3a1'>{cuda_status}</font>")
        perf_layout.addWidget(gpu_label)
        
        self.lbl_fps = QLabel("<b>İşleme Hızı (FPS):</b> <font color='#89b4fa'>0 FPS</font>")
        self.lbl_stream_source = QLabel("<b>Aktif Kaynak:</b> <font color='#6c7086'>Hazır (Yayın Yok)</font>")
        
        perf_layout.addWidget(self.lbl_fps)
        perf_layout.addWidget(self.lbl_stream_source)
        
        left_panel.addWidget(perf_frame, 2)
        
        # 5. KONTROL VE MEDYA SEÇİM BUTONLARI
        self.btn_open = QPushButton("📸 Resim / Video Yükle")
        self.btn_open.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa; color: #11111b; font-weight: bold; font-size: 15px;
                border-radius: 8px; padding: 12px;
            }
            QPushButton:hover { background-color: #b4befe; }
        """)
        self.btn_open.clicked.connect(self.open_file)
        left_panel.addWidget(self.btn_open, 1)

        self.btn_camera = QPushButton("📹 Canlı Kamera Başlat")
        self.btn_camera.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1; color: #11111b; font-weight: bold; font-size: 15px;
                border-radius: 8px; padding: 12px;
            }
            QPushButton:hover { background-color: #94e2d5; }
        """)
        self.btn_camera.clicked.connect(self.toggle_camera)
        left_panel.addWidget(self.btn_camera, 1)

        # ================= SAĞ PANEL (GÖRÜNTÜ ALANI) =================
        right_panel = QVBoxLayout()
        self.video_label = QLabel("Lütfen test etmek için bir görsel, video seçin veya canlı kamerayı başlatın.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            border: 2px dashed #45475a; border-radius: 12px; 
            background-color: #11111b; font-size: 16px; color: #6c7086;
        """)
        self.video_label.setMinimumSize(850, 650)
        right_panel.addWidget(self.video_label)
        
        main_layout.addLayout(left_panel, 32)
        main_layout.addLayout(right_panel, 68)
        self.setLayout(main_layout)

    def stop_current_stream(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_webcam = False
        self.btn_camera.setText("📹 Canlı Kamera Başlat")
        self.btn_camera.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1; color: #11111b; font-weight: bold; font-size: 15px;
                border-radius: 8px; padding: 12px;
            }
            QPushButton:hover { background-color: #94e2d5; }
        """)

    def open_file(self):
        self.stop_current_stream()
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Medya Dosyası Seç", "", "Medya Dosyaları (*.jpg *.jpeg *.png *.mp4 *.avi)"
        )
        if file_path:
            if file_path.lower().endswith(('.mp4', '.avi')):
                self.lbl_stream_source.setText("<b>Aktif Kaynak:</b> <font color='#f9e2af'>Video Dosyası</font>")
                self.start_video_stream(file_path)
            else:
                self.lbl_stream_source.setText("<b>Aktif Kaynak:</b> <font color='#f9e2af'>Statik Resim</font>")
                self.process_image(file_path)

    def toggle_camera(self):
        if self.is_webcam:
            self.stop_current_stream()
            self.lbl_stream_source.setText("<b>Aktif Kaynak:</b> <font color='#6c7086'>Kamera Durduruldu</font>")
        else:
            self.stop_current_stream()
            self.is_webcam = True
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.video_label.setText("❌ Kamera Akışı Başlatılamadı! Lütfen Kameranızı Kontrol Edin.")
                self.is_webcam = False
                return
            
            self.btn_camera.setText("🛑 Kamerayı Durdur")
            self.btn_camera.setStyleSheet("""
                QPushButton {
                    background-color: #f38ba8; color: #11111b; font-weight: bold; font-size: 15px;
                    border-radius: 8px; padding: 12px;
                }
                QPushButton:hover { background-color: #eba0ac; }
            """)
            self.lbl_stream_source.setText("<b>Aktif Kaynak:</b> <font color='#a6e3a1'>🔴 Canlı Kamera (Webcam)</font>")
            self.timer.start(30)

    def analyze_and_draw(self, frame_or_path):
        curr_time = time.time()
        fps = 1.0 / (curr_time - self.prev_time + 1e-6)
        self.prev_time = curr_time
        self.lbl_fps.setText(f"<b>İşleme Hızı (FPS):</b> <font color='#89b4fa'>{fps:.1f} FPS</font>")

        results = self.model.predict(source=frame_or_path, imgsz=800, conf=0.25, verbose=False)
        result = results[0]
        
        if isinstance(frame_or_path, str):
            annotated_frame = cv2.imread(frame_or_path)
        else:
            annotated_frame = frame_or_path.copy()

        bird_count = 0
        human_count = 0
        animal_count = 0
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                if cls_id == 1:  # Kuş
                    bird_count += 1
                    color = (0, 255, 255)  # Sarı
                    display_text = f"kus {conf:.2f}"
                elif cls_id == 0:  # İnsan
                    human_count += 1
                    color = (255, 0, 0)    # Mavi
                    display_text = f"insan {conf:.2f}"
                else:  # Hayvan (cls_id == 2)
                    animal_count += 1
                    color = (0, 255, 0)    # Yeşil
                    display_text = f"hayvan {conf:.2f}"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                (w, h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, max(0, y1 - 25)), (x1 + w, max(0, y1)), color, -1)
                cv2.putText(annotated_frame, display_text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            if bird_count >= 5:
                self.alert_label.setText(f"<b>Sürü Durumu:</b> <font color='#f38ba8'>⚠️ KUŞ SÜRÜSÜ ALGILANDI ({bird_count} Kuş)</font>")
                cv2.putText(annotated_frame, f"ALARM: KUS SURUSU ({bird_count})", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                self.alert_label.setText(f"<b>Sürü Durumu:</b> <font color='#a6e3a1'>Normal ({bird_count} Kuş)</font>")
        else:
            self.alert_label.setText("<b>Sürü Durumu:</b> <font color='#a6e3a1'>Normal (0 Kuş)</font>")

        self.lbl_human_count.setText(f"<b>• İnsan Tespiti:</b> <font color='#89b4fa'>{human_count}</font>")
        self.lbl_bird_count.setText(f"<b>• Kuş Tespiti:</b> <font color='#f9e2af'>{bird_count}</font>")
        self.lbl_animal_count.setText(f"<b>• Hayvan Tespiti:</b> <font color='#a6e3a1'>{animal_count}</font>")

        return annotated_frame

    def process_image(self, img_path):
        annotated_frame = self.analyze_and_draw(img_path)
        self.display_frame(annotated_frame)

    def start_video_stream(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.timer.start(30)

    def update_video_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                annotated_frame = self.analyze_and_draw(frame)
                self.display_frame(annotated_frame)
            else:
                self.stop_current_stream()
                self.lbl_stream_source.setText("<b>Aktif Kaynak:</b> <font color='#6c7086'>Video Bitti</font>")

    def display_frame(self, frame_bgr):
        rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DroneDetectionApp()
    ex.show()
    sys.exit(app.exec_())
    