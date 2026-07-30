import os
import pandas as pd
import matplotlib.pyplot as plt

# Eğitim çıktısının kaydedildiği klasör ve CSV yolu
RUN_DIR = os.path.join("runs", "detect", "drone_yolov8_s_run")
CSV_PATH = os.path.join(RUN_DIR, "results.csv")

def make_custom_plots():
    if not os.path.exists(CSV_PATH):
        print(f"Hata: {CSV_PATH} bulunamadı! Lütfen eğitim klasör yolunu kontrol et.")
        return

    # Sütun isimlerindeki gizli boşlukları temizleyelim
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    epochs = df['epoch']

    # --- 1. SENİN İSTEDİĞİN 'test/' BAŞLIKLI 10'LU LOSS & METRİK GRAFİĞİ ---
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.tight_layout(pad=3.0)

    # 1. Satır: Loss ve Metrikler
    axes[0, 0].plot(epochs, df['train/box_loss'], label='results', color='#1f77b4', marker='o', ms=2)
    axes[0, 0].set_title('test/box_loss')

    axes[0, 1].plot(epochs, df['train/cls_loss'], label='results', color='#1f77b4', marker='o', ms=2)
    axes[0, 1].set_title('test/cls_loss')

    axes[0, 2].plot(epochs, df['train/dfl_loss'], label='results', color='#1f77b4', marker='o', ms=2)
    axes[0, 2].set_title('test/dfl_loss')

    axes[0, 3].plot(epochs, df['metrics/precision(B)'], label='results', color='#1f77b4', marker='o', ms=2)
    axes[0, 3].set_title('metrics/precision(B)')

    axes[0, 4].plot(epochs, df['metrics/recall(B)'], label='results', color='#1f77b4', marker='o', ms=2)
    axes[0, 4].set_title('metrics/recall(B)')

    # 2. Satır: Val Loss'lar (İstediğin gibi test/ ismiyle) ve mAP değerleri
    axes[1, 0].plot(epochs, df['val/box_loss'], label='results', color='#1f77b4', marker='o', ms=2)
    axes[1, 0].set_title('test/val_box_loss')

    axes[1, 1].plot(epochs, df['val/cls_loss'], label='results', color='#1f77b4', marker='o', ms=2)
    axes[1, 1].set_title('test/val_cls_loss')

    axes[1, 2].plot(epochs, df['val/dfl_loss'], label='results', color='#1f77b4', marker='o', ms=2)
    axes[1, 2].set_title('test/val_dfl_loss')

    axes[1, 3].plot(epochs, df['metrics/mAP50(B)'], label='results', color='#1f77b4', marker='o', ms=2)
    axes[1, 3].set_title('metrics/mAP50(B)')

    axes[1, 4].plot(epochs, df['metrics/mAP50-95(B)'], label='results', color='#1f77b4', marker='o', ms=2)
    axes[1, 4].set_title('metrics/mAP50-95(B)')

    output_1 = os.path.join(RUN_DIR, "custom_test_results.png")
    plt.savefig(output_1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Ana grafik kaydedildi: {output_1}")

    # --- 2. ACCURACY & INTERSECTION OVER UNION (IoU) GRAFİĞİ ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    # Accuracy Temsili (Precision ve Recall Ortalaması)
    accuracy = (df['metrics/precision(B)'] + df['metrics/recall(B)']) / 2
    ax[0].plot(epochs, accuracy, color='purple', marker='o', ms=2)
    ax[0].set_title('test/accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].grid(True, linestyle='--', alpha=0.6)

    # IoU (mAP50)
    ax[1].plot(epochs, df['metrics/mAP50(B)'], color='green', marker='o', ms=2)
    ax[1].set_title('test/intersection_over_union (IoU@0.50)')
    ax[1].set_xlabel('Epoch')
    ax[1].grid(True, linestyle='--', alpha=0.6)

    output_2 = os.path.join(RUN_DIR, "accuracy_iou_metrics.png")
    plt.savefig(output_2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Accuracy & IoU grafiği kaydedildi: {output_2}")

if __name__ == "__main__":
    make_custom_plots()