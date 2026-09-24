import os
import json
import glob
import random
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ==========================================
#               НАСТРОЙКИ
# ==========================================
METADATA_FILE = 'DATASTORE/wines_metadata.json'  # Файл для связи ID файлов и названий папок (slug)
INPUT_DIR = Path('DATASTORE/wine_png')      # Папка с исходными изображениями (как в scrapper)
OUTPUT_DIR = Path('DATASTORE/dataset_arc')  # Итоговая папка для обучения (как в generate_train_val)
YOLO_MODEL_PATH = 'models/YOLO_best_100.pt'      # Ваши веса сегментации

TARGET_SIZE = 640                      # Размер квадратного тензора
BG_COLOR = (128, 128, 128)             # Нейтральный серый фон

TRAIN_VARIANTS = 35                    # Кол-во 3D синтетики для обучения
VAL_VARIANTS = 10                      # Кол-во 3D синтетики для экзамена


def apply_clahe(img_bgr):
    """Вытягивает контраст текста, не искажая исходные цвета."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def smart_letterbox(img, target_size, bg_color):
    """Умный паддинг: INTER_AREA для сжатия, INTER_CUBIC для увеличения (сохраняет резкость)."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Защита от потери качества при изменении размера
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    
    canvas = np.full((target_size, target_size, 3), bg_color, dtype=np.uint8)
    x_off, y_off = (target_size - new_w) // 2, (target_size - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def extract_label_yolo(img, yolo_model):
    """Вырезает этикетку по точной маске YOLO. Всё за контуром становится прозрачным (Alpha=0)."""
    results = yolo_model.predict(source=img, conf=0.5, retina_masks=True, verbose=False)
    
    # Если YOLO не нашел этикетку, возвращаем картинку как есть, добавив альфа-канал
    if not results or len(results[0].boxes) == 0 or results[0].masks is None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) if img.shape[2] == 3 else img

    result = results[0]
    best_idx = np.argmax(result.boxes.conf.cpu().numpy())
    
    box = result.boxes.xyxy[best_idx].cpu().numpy().astype(int)
    polygon = result.masks.xy[best_idx].astype(np.int32)

    # Добавляем альфа-канал
    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # Создаем черную маску и заливаем полигон белым
    blank_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(blank_mask, [polygon], 255)

    # Применяем маску к альфа-каналу
    img_bgra[:, :, 3] = blank_mask
    
    # Обрезаем лишнюю прозрачную пустоту по Bounding Box
    x1, y1, x2, y2 = box
    return img_bgra[y1:y2, x1:x2]


class FastBottleSynthesizer:
    def __init__(self, target_size=640, bg_color=(128, 128, 128)):
        self.target_size = target_size
        self.bg_color = np.array(bg_color, dtype=np.uint8)

    def synthesize_variant(self, img_bgra):
        """Векторная генерация 3D искажений с освещением Blinn-Phong."""
        h, w = img_bgra.shape[:2]

        fov_deg = random.uniform(115, 155)
        theta_max = np.deg2rad(fov_deg / 2.0)
        R = (w / 2.0) / theta_max
        pitch_amplitude = random.uniform(-0.12, 0.12) * h
        yaw_offset = random.uniform(-0.35, 0.35) * theta_max

        light_x = random.uniform(-1.0, 1.0)
        shininess = random.uniform(35.0, 85.0)
        glare_intensity = random.uniform(80.0, 180.0)

        w_out = int(2 * R * np.sin(theta_max))
        h_out = h + int(abs(pitch_amplitude))

        x_indices = np.arange(w_out, dtype=np.float32)
        val = np.clip((x_indices - w_out / 2.0) / R, -1.0, 1.0)
        theta = np.arcsin(val)

        theta_src = theta + yaw_offset
        valid_mask_1d = np.abs(theta_src) <= theta_max

        x_src_1d = np.where(valid_mask_1d, (theta_src / theta_max) * (w / 2.0) + (w / 2.0), -1.0)
        map_x = np.tile(x_src_1d, (h_out, 1)).astype(np.float32)
        alpha_mask = np.tile(valid_mask_1d.astype(np.float32), (h_out, 1))

        normals_x = np.tile(np.sin(theta), (h_out, 1)).astype(np.float32)
        normals_z = np.tile(np.cos(theta), (h_out, 1)).astype(np.float32)

        y_curve_offset = pitch_amplitude * (np.cos(theta) - 1.0)
        y_indices = np.arange(h_out, dtype=np.float32)[:, np.newaxis]
        map_y = (y_indices - max(0, pitch_amplitude) - y_curve_offset).astype(np.float32)

        # INTER_CUBIC сохраняет четкость букв при 3D-выгибании
        warped = cv2.remap(img_bgra, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        bgr = warped[:, :, :3].astype(np.float32)
        alpha = (warped[:, :, 3].astype(np.float32) / 255.0) * alpha_mask

        light_dir = np.array([light_x, 0.2, 0.8], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        half_vec = light_dir + view_dir
        half_vec /= np.linalg.norm(half_vec)

        diffuse = np.clip(normals_x * light_dir[0] + normals_z * light_dir[2], 0.0, 1.0)
        ambient = 0.5 
        shading = ambient + (1.0 - ambient) * diffuse

        spec_dot = np.clip(normals_x * half_vec[0] + normals_z * half_vec[2], 0.0, 1.0)
        specular = np.power(spec_dot, shininess) * glare_intensity

        shading_3d = np.stack([shading] * 3, axis=2)
        specular_3d = np.stack([specular] * 3, axis=2)

        final_bgr = np.clip((bgr * shading_3d) + specular_3d, 0.0, 255.0)

        # Добавляем немного матричного шума (как на камерах телефонов)
        noise = np.random.normal(0, random.uniform(1.0, 3.0), final_bgr.shape).astype(np.float32)
        final_bgr = np.clip(final_bgr + noise, 0.0, 255.0)

        # Сплющиваем прозрачную альфа-маску 3D-этикетки на сплошной серый фон
        mask_3d = np.stack([alpha], axis=2)
        bg = np.full_like(final_bgr, self.bg_color, dtype=np.float32)
        blended = (final_bgr * mask_3d + bg * (1.0 - mask_3d)).astype(np.uint8)

        return smart_letterbox(blended, self.target_size, self.bg_color)


def main():
    print("Загрузка метаданных и модели YOLO...")
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    yolo_model = YOLO(YOLO_MODEL_PATH)
    synthesizer = FastBottleSynthesizer(target_size=TARGET_SIZE, bg_color=BG_COLOR)
    
    (OUTPUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val").mkdir(parents=True, exist_ok=True)

    processed_count = 0

    for item in tqdm(metadata, desc="Пайплайн: YOLO -> CLAHE -> 3D Synth"):
        scraped_id = item.get("scraped_id")
        slug = item.get("raw_metadata", {}).get("slug")
        if not scraped_id or not slug: 
            continue
            
        # Поддерживаем любые форматы и окончания имен файлов (wine_12.jpg, wine_12_label.webp)
        search_pattern = os.path.join(INPUT_DIR, f"wine_{scraped_id}_*.*")
        found_files = glob.glob(search_pattern)
        if not found_files:
             found_files = glob.glob(os.path.join(INPUT_DIR, f"wine_{scraped_id}.*"))
             
        if not found_files: 
            continue
            
        img = cv2.imread(found_files[0])
        if img is None: 
            continue
            
        # 1. Точная обрезка по маске (фон становится прозрачным)
        img_bgra = extract_label_yolo(img, yolo_model)
        
        # 2. Вытягиваем контраст (CLAHE)
        bgr_clahe = apply_clahe(img_bgra[:, :, :3])
        img_bgra_clahe = np.dstack((bgr_clahe, img_bgra[:, :, 3]))

        train_slug_dir = OUTPUT_DIR / "train" / slug
        val_slug_dir = OUTPUT_DIR / "val" / slug
        train_slug_dir.mkdir(parents=True, exist_ok=True)
        val_slug_dir.mkdir(parents=True, exist_ok=True)

        # 3. Сохранение идеала (на сером фоне)
        alpha = (img_bgra_clahe[:, :, 3] / 255.0)[:, :, np.newaxis]
        gray_bg = np.full_like(bgr_clahe, BG_COLOR, dtype=np.uint8)
        flattened_ideal = (bgr_clahe * alpha + gray_bg * (1.0 - alpha)).astype(np.uint8)
        
        ideal_canvas = smart_letterbox(flattened_ideal, TARGET_SIZE, BG_COLOR)
        cv2.imwrite(str(train_slug_dir / "ideal.jpg"), ideal_canvas, [cv2.IMWRITE_JPEG_QUALITY, 99])

        # 4. Генерация 3D (Train + Val)
        for i in range(TRAIN_VARIANTS):
            synth_img = synthesizer.synthesize_variant(img_bgra_clahe)
            cv2.imwrite(str(train_slug_dir / f"synth_{i:02d}.jpg"), synth_img, [cv2.IMWRITE_JPEG_QUALITY, 96])

        for i in range(VAL_VARIANTS):
            synth_img = synthesizer.synthesize_variant(img_bgra_clahe)
            cv2.imwrite(str(val_slug_dir / f"val_synth_{i:02d}.jpg"), synth_img, [cv2.IMWRITE_JPEG_QUALITY, 96])
            
        processed_count += 1

    print(f"\n✅ Готово! Обработано {processed_count} вин.")
    print(f"📁 Датасет готов к обучению в папке: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()