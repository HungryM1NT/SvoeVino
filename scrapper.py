import os
import cv2
import numpy as np
from ultralytics import YOLO

# --- Настройки путей ---
INPUT_DIR = 'tw1'
OUTPUT_DIR = 'tws1'
MODEL_PATH = 'best_100_2.pt'

def crop_labels_masks():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]

    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        # Предсказание с извлечением масок
        results = model.predict(source=img, conf=0.5, retina_masks=True, verbose=False)
        result = results[0]

        if result.masks is None or len(result.masks) == 0:
            continue

        base_name = os.path.splitext(img_name)[0]

        # Добавляем альфа-канал к исходному изображению (для прозрачности)
        # OpenCV читает в BGR, переводим в BGRA
        img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # Перебираем все маски и их координаты
        for i, (mask, box) in enumerate(zip(result.masks.xy, result.boxes.xyxy)):
            # 1. Создаем пустую (черную) маску размером с исходное фото
            blank_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            # 2. Рисуем белый полигон сегментации на черной маске
            polygon = np.array(mask, dtype=np.int32)
            cv2.fillPoly(blank_mask, [polygon], 255)

            # 3. Применяем маску к альфа-каналу изображения
            isolated_label = img_bgra.copy()
            isolated_label[:, :, 3] = blank_mask # Всё за пределами маски станет прозрачным

            # 4. Обрезаем картинку по bounding box, чтобы не сохранять много пустого места
            x1, y1, x2, y2 = map(int, box.tolist())
            cropped_transparent_label = isolated_label[y1:y2, x1:x2]

            # 5. Сохраняем в PNG (обязательно PNG для сохранения прозрачности)
            save_path = os.path.join(OUTPUT_DIR, f"{base_name}_label_{i+1}_mask.png")
            cv2.imwrite(save_path, cropped_transparent_label)

    print("Готово! Этикетки вырезаны по контуру и сохранены в", OUTPUT_DIR)

if __name__ == "__main__":
    crop_labels_masks()