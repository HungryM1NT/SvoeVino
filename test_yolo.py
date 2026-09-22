import cv2
from ultralytics import YOLO

def run_yolo_segmentation(image_path, output_path="result.jpg"):
    # Загружаем предобученную модель YOLOv8 для сегментации. 
    # Суффикс '-seg' означает, что модель обучена именно на сегментацию, а не просто на рамки (bounding boxes).
    print("Загрузка модели...")
    model = YOLO("best_100_2.pt")

    # Запускаем инференс (распознавание)
    # Параметр conf=0.5 означает, что мы берем только объекты с уверенностью >= 50%
    print(f"Обработка изображения: {image_path}")
    results = model.predict(source=image_path, conf=0.5)

    # YOLO возвращает список результатов (для каждого кадра/изображения)
    # Так как у нас одно изображение, берем первый элемент
    result = results[0]

    # result.plot() автоматически рисует маски, рамки и подписи классов поверх оригинального фото
    annotated_image = result.plot()

    # Сохраняем результат на диск
    cv2.imwrite(output_path, annotated_image)
    print(f"Готово! Результат сохранен в {output_path}")

    # (Опционально) Показываем результат в окне
    cv2.imshow("YOLO Segmentation", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Укажи путь к твоему изображению
    # Если файла нет, скрипт выдаст ошибку, поэтому не забудь подставить реальный путь
    IMAGE_PATH = "p1.jpg" 
    
    run_yolo_segmentation(IMAGE_PATH)