import os
import json
import glob
import cv2
import numpy as np

# --- Настройки ---
METADATA_FILE = 'wines_metadata.json'
SOURCE_DIR = 'wine_scrapped'
TARGET_DIR = 'dataset/train'
TARGET_SIZE = 640  # Стандартный размер для YOLO и EfficientNet
BG_COLOR = (128, 128, 128)  # Серый фон (B, G, R в OpenCV)

def apply_clahe(img):
    """
    Применяет CLAHE к L-каналу в цветовом пространстве LAB, 
    чтобы улучшить контраст, не искажая исходные цвета этикетки.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Настройки CLAHE (clipLimit 2.0 или 3.0 обычно дают лучший результат)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def letterbox_image(img, target_size, bg_color):
    """
    Вписывает изображение в квадрат target_size x target_size 
    с сохранением пропорций и заливкой фона.
    """
    h, w = img.shape[:2]
    
    # Вычисляем коэффициент масштабирования
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Меняем размер оригинальной картинки
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Создаем серый холст
    canvas = np.full((target_size, target_size, 3), bg_color, dtype=np.uint8)
    
    # Вычисляем отступы, чтобы разместить картинку по центру
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - target_size) // 2 
    y_offset = (target_size - new_h) // 2
    
    # Вставляем картинку в центр холста
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    return canvas

def main():
    # Загружаем метаданные
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    processed_count = 0
    missing_count = 0

    for item in metadata:
        scraped_id = item.get("scraped_id")
        raw_meta = item.get("raw_metadata", {})
        slug = raw_meta.get("slug")
        
        if not scraped_id or not slug:
            continue
            
        # Ищем исходный файл по маске wine_{id}_* (поддерживает любой хвост и расширение)
        search_pattern = os.path.join(SOURCE_DIR, f"wine_{scraped_id}_*.*")
        found_files = glob.glob(search_pattern)
        
        # Если файлы вдруг называются просто wine_1.webp, проверим и это
        if not found_files:
             search_pattern_alt = os.path.join(SOURCE_DIR, f"wine_{scraped_id}.*")
             found_files = glob.glob(search_pattern_alt)
        
        if not found_files:
            missing_count += 1
            print(f"[ПРОПУСК] Фото для ID {scraped_id} ({slug}) не найдено.")
            continue
            
        source_img_path = found_files[0]
        
        # Читаем изображение
        img = cv2.imread(source_img_path)
        if img is None:
            print(f"[ОШИБКА ЧТЕНИЯ] Не удалось прочитать {source_img_path}")
            continue
            
        # 1. Применяем CLAHE
        img_clahe = apply_clahe(img)
        
        # 2. Делаем Letterboxing с серым фоном
        img_final = letterbox_image(img_clahe, TARGET_SIZE, BG_COLOR)
        
        # Создаем папку класса (slug)
        class_dir = os.path.join(TARGET_DIR, slug)
        os.makedirs(class_dir, exist_ok=True)
        
        # 3. Сохраняем в формате JPG (качество 95 для ML)
        target_filename = f"wine_{scraped_id}.jpg"
        target_path = os.path.join(class_dir, target_filename)
        
        cv2.imwrite(target_path, img_final, [cv2.IMWRITE_JPEG_QUALITY, 95])
        processed_count += 1
        
    print("-" * 30)
    print(f"Готово! Успешно обработано: {processed_count} изображений.")
    if missing_count > 0:
        print(f"Не найдено исходников для: {missing_count} вин.")

if __name__ == "__main__":
    main()