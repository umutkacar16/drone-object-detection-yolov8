import os
import zipfile
import shutil
import glob
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISDRONE_ZIP = os.path.join(BASE_DIR, "visdrone.zip")
ANIMALS_ZIP = os.path.join(BASE_DIR, "animals.zip")

EXTRACT_DIR = os.path.join(BASE_DIR, "extracted_temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "merged_dataset")

# HEDEF SINIFLAR (data.yaml ile tam uyumlu):
# 0: insan
# 1: kus
# 2: hayvan

# VisDrone Orijinal: 1 -> Pedestrian, 2 -> People
VISDRONE_MAPPING = {1: 0, 2: 0}

def setup_folders():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

def extract_zips():
    print("-> Zip dosyaları ayıklanıyor...")
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
        
    for name, zip_path in [("VisDrone", VISDRONE_ZIP), ("Animals", ANIMALS_ZIP)]:
        if os.path.exists(zip_path):
            target_extract = os.path.join(EXTRACT_DIR, name)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_extract)
            print(f"   {name} başarıyla ayıklandı.")
        else:
            print(f"[UYARI] {zip_path} bulunamadı!")

def convert_visdrone_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    x_center = box[0] + box[2] / 2.0
    y_center = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    
    return (x_center * dw, y_center * dh, w * dw, h * dh)

def find_matching_image(txt_path):
    base_dir = os.path.dirname(txt_path)
    possible_img_dirs = [
        base_dir.replace("annotations", "images").replace("labels", "images").replace("Labels", "Images"),
        base_dir
    ]
    file_name = os.path.splitext(os.path.basename(txt_path))[0]
    extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    
    for img_dir in possible_img_dirs:
        if os.path.exists(img_dir):
            for ext in extensions:
                img_path = os.path.join(img_dir, file_name + ext)
                if os.path.exists(img_path):
                    return img_path
    return None

def process_visdrone():
    print("-> VisDrone verileri işleniyor...")
    source_path = os.path.join(EXTRACT_DIR, "VisDrone")
    if not os.path.exists(source_path):
        return

    all_txt_files = glob.glob(os.path.join(source_path, "**", "*.txt"), recursive=True)
    copied_count = 0
    
    for txt_file in all_txt_files:
        if "classes.txt" in txt_file:
            continue
            
        img_path = find_matching_image(txt_file)
        if not img_path:
            continue
            
        file_path_lower = txt_file.lower()
        split = 'val' if any(x in file_path_lower for x in ['val', 'test', 'valid']) else 'train'
        
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception:
            continue
            
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().replace(',', ' ').split()
            if len(parts) < 8:
                continue
                
            try:
                cls_id = int(parts[5])
                score = int(parts[4])
                if score == 0:
                    continue
            except ValueError:
                continue
                
            if cls_id in VISDRONE_MAPPING:
                new_cls = VISDRONE_MAPPING[cls_id]
                x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                
                if w <= 0 or h <= 0:
                    continue
                    
                nx, ny, nw, nh = convert_visdrone_bbox((img_w, img_h), (x, y, w, h))
                new_lines.append(f"{new_cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
                
        if new_lines:
            base_name = f"VisDrone_{os.path.basename(txt_file)}"
            out_label_path = os.path.join(OUTPUT_DIR, split, 'labels', base_name)
            
            with open(out_label_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
                
            img_ext = os.path.splitext(img_path)[1]
            out_img_name = f"VisDrone_{os.path.splitext(os.path.basename(txt_file))[0]}{img_ext}"
            dest_img_path = os.path.join(OUTPUT_DIR, split, 'images', out_img_name)
            shutil.copy(img_path, dest_img_path)
            copied_count += 1
            
    print(f"   VisDrone setinden {copied_count} adet resim-etiket ikilisi aktarıldı.")

def process_animals():
    print("-> Animals verileri (Birds ve WAID) işleniyor...")
    source_path = os.path.join(EXTRACT_DIR, "Animals")
    if not os.path.exists(source_path):
        return

    all_txt_files = glob.glob(os.path.join(source_path, "**", "*.txt"), recursive=True)
    copied_count = 0
    
    for txt_file in all_txt_files:
        if "classes.txt" in txt_file:
            continue
            
        img_path = find_matching_image(txt_file)
        if not img_path:
            continue
            
        file_path_lower = txt_file.lower()
        split = 'val' if any(x in file_path_lower for x in ['val', 'test', 'valid']) else 'train'
        
        # Dosya yolunda "birds" geçiyorsa -> 1 (kus), WAID-final geçiyorsa -> 2 (hayvan)
        if "bird" in file_path_lower:
            target_class = 1  # kus
        else:
            target_class = 2  # hayvan
        
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            # Etiketin orijinal sınıf ID'sini ezip hedef sınıfımızı yazıyoruz
            parts[0] = str(target_class)
            new_lines.append(" ".join(parts[:5]) + "\n")
                
        if new_lines:
            base_name = f"Animals_{os.path.basename(txt_file)}"
            out_label_path = os.path.join(OUTPUT_DIR, split, 'labels', base_name)
            
            with open(out_label_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
                
            img_ext = os.path.splitext(img_path)[1]
            out_img_name = f"Animals_{os.path.splitext(os.path.basename(txt_file))[0]}{img_ext}"
            dest_img_path = os.path.join(OUTPUT_DIR, split, 'images', out_img_name)
            shutil.copy(img_path, dest_img_path)
            copied_count += 1
            
    print(f"   Animals setinden {copied_count} adet resim-etiket ikilisi aktarıldı.")

if __name__ == "__main__":
    setup_folders()
    extract_zips()
    process_visdrone()
    process_animals()
    
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
        
    print(f"\n[TAMAMLANDI] Veri seti hazırlığı bitti! Konum: {OUTPUT_DIR}")
    