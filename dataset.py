import os
import zipfile
import shutil
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISDRONE_ZIP = os.path.join(BASE_DIR, "visdrone.zip")
ANIMALS_ZIP = os.path.join(BASE_DIR, "animals.zip")

EXTRACT_DIR = os.path.join(BASE_DIR, "extracted_temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "merged_dataset")

# Sınıf Eşlemeleri (0: insan, 1: kus, 2: evcil_hayvan, 3: vahsi_hayvan)
VISDRONE_MAPPING = {1: 0, 2: 0}
ANIMAL_CLASS_MAPPING = {0: 2, 1: 2, 2: 3, 3: 1}

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

def convert_labels(label_path, mapping, output_path):
    with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            parts = line.strip().split(',')
        if not parts or len(parts) < 5:
            continue
            
        try:
            old_cls = int(parts[0])
        except ValueError:
            continue
            
        if old_cls in mapping:
            parts[0] = str(mapping[old_cls])
            new_lines.append(" ".join(parts[:5]) + "\n")
            
    if new_lines:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    return False

def find_matching_image(txt_path):
    base_dir = os.path.dirname(txt_path)
    possible_img_dirs = [
        base_dir.replace("labels", "images").replace("Labels", "Images"),
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

def process_dataset(source_name, mapping):
    print(f"-> {source_name} verileri işleniyor...")
    source_path = os.path.join(EXTRACT_DIR, source_name)
    if not os.path.exists(source_path):
        return

    all_txt_files = glob.glob(os.path.join(source_path, "**", "*.txt"), recursive=True)
    copied_count = 0
    
    for txt_file in all_txt_files:
        if "classes.txt" in txt_file:
            continue
            
        img_path = find_matching_image(txt_file)
        if img_path:
            file_path_lower = txt_file.lower()
            split = 'val' if any(x in file_path_lower for x in ['val', 'test', 'valid']) else 'train'
            
            base_name = os.path.basename(txt_file)
            out_label_path = os.path.join(OUTPUT_DIR, split, 'labels', base_name)
            
            if convert_labels(txt_file, mapping, out_label_path):
                dest_img_path = os.path.join(OUTPUT_DIR, split, 'images', os.path.basename(img_path))
                shutil.copy(img_path, dest_img_path)
                copied_count += 1
                
    print(f"   {source_name} setinden {copied_count} adet resim-etiket ikilisi aktarıldı.")

if __name__ == "__main__":
    setup_folders()
    extract_zips()
    process_dataset("VisDrone", VISDRONE_MAPPING)
    process_dataset("Animals", ANIMAL_CLASS_MAPPING)
    
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
        
    print(f"\n[TAMAMLANDI] Veri seti hazırlığı bitti! Konum: {OUTPUT_DIR}")
