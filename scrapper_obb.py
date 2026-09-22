import os
import cv2
from ultralytics import YOLO

# --- Настройки путей ---
INPUT_DIR = 'tw1'
OUTPUT_DIR = 'tws1obb'
MODEL_PATH = 'best_100_2.pt'  # Замените на путь к вашей обученной модели!

def crop_labels_bbox():
    # Создаем папку для сохранения, если ее нет
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Загружаем модель
    print("Загрузка модели...")
    model = YOLO(MODEL_PATH)

    # Получаем список всех изображений в папке
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]

    print(f"Найдено {len(images)} изображений. Начинаем обработку...")

    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Не удалось прочитать изображение: {img_name}")
            continue

        # Запускаем предсказание
        results = model.predict(source=img, conf=0.5, verbose=False) # conf=0.5 - порог уверенности (можно менять)

        # Берем первый результат (так как передаем по одной картинке)
        result = results[0]
        
        # Если ничего не найдено, пропускаем
        if result.boxes is None or len(result.boxes) == 0:
            print(f"Этикетки не найдены на {img_name}")
            continue

        base_name = os.path.splitext(img_name)[0]

        # Проходим по всем найденным объектам на фото
        for i, box in enumerate(result.boxes):
            # Получаем координаты прямоугольника
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Вырезаем этикетку
            cropped_label = img[y1:y2, x1:x2]

            # Сохраняем
            save_path = os.path.join(OUTPUT_DIR, f"{base_name}_label_{i+1}.png")
            cv2.imwrite(save_path, cropped_label)
            
    print("Готово! Все этикетки сохранены в папку", OUTPUT_DIR)

if __name__ == "__main__":
    crop_labels_bbox()