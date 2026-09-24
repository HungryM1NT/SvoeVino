import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
from ultralytics import YOLO
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

class InferenceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.neck = nn.Sequential(
            nn.Linear(self.backbone.num_features, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.neck(features)
        return F.normalize(embeddings, p=2, dim=1)


class WineRecognizer:
    def __init__(self, yolo_weights, metric_weights, qdrant_path="qdrant_db", collection_name="wines"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Инициализация на {self.device}...")

        self.yolo = YOLO(yolo_weights)
        self.embedder = InferenceNet(embedding_size=512).to(self.device)
        checkpoint = torch.load(metric_weights, map_location=self.device)
        self.embedder.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.embedder.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.qdrant = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name

    def _apply_clahe(self, img_bgr):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _smart_letterbox(self, img, target_size=224, bg_color=(128, 128, 128)):
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        # Выбираем алгоритм для сохранения четкости
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        
        canvas = np.full((target_size, target_size, 3), bg_color, dtype=np.uint8)
        x_off, y_off = (target_size - new_w) // 2, (target_size - new_h) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        return canvas

    def recognize(self, image_path, expected_slug=None):
        results = self.yolo.predict(image_path, verbose=False, conf=0.5)
        
        if len(results[0].boxes) == 0 or results[0].masks is None:
            return None, None, None 
            
        confidences = results[0].boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)

        box = results[0].boxes.xyxy[best_idx].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        polygon = results[0].masks.xy[best_idx].astype(np.int32)

        orig_img = cv2.imread(image_path)
        if orig_img is None: return None, None, None

        # 1. Вырезаем "сырую" этикетку по BBox
        raw_crop = orig_img[y1:y2, x1:x2]
        
        # 2. Применяем CLAHE только к оригинальным пикселям
        clahe_crop = self._apply_clahe(raw_crop)

        # 3. Смещаем координаты полигона относительно вырезанного куска
        shifted_polygon = polygon - np.array([x1, y1])

        # 4. Накладываем идеальный серый фон строго за пределами полигона
        bg_color = (128, 128, 128)
        masked_img = np.full_like(clahe_crop, bg_color, dtype=np.uint8)
        binary_mask = np.zeros(clahe_crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(binary_mask, [shifted_polygon], 255)
        
        mask_boolean = binary_mask[:, :, np.newaxis] == 255
        crop_final = np.where(mask_boolean, clahe_crop, masked_img)

        # 5. Умный ресайз
        processed_img = self._smart_letterbox(crop_final, target_size=224) 

        # Конвертируем для PyTorch
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                embedding = self.embedder(tensor).cpu().numpy()[0].tolist()

        top_results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=5
        ).points
        
        expected_score = None
        if expected_slug:
            target_response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=embedding,
                query_filter=Filter(
                    must=[FieldCondition(key="slug", match=MatchValue(value=expected_slug))]
                ),
                limit=1
            )
            if target_response.points:
                expected_score = target_response.points[0].score

        return top_results, expected_score, crop_final


if __name__ == "__main__":
    YOLO_WEIGHTS = "models/YOLO_best_100.pt"
    METRIC_WEIGHTS = "models/ArcFace_v2.pth"
    TEST_IMAGE = "96.66_22-08-2026_20-44-40.webp"
    
    CORRECT_SLUG = "perovskih_polusuhoe_krasnoe"

    recognizer = WineRecognizer(
        yolo_weights=YOLO_WEIGHTS, 
        metric_weights=METRIC_WEIGHTS
    )
    
    top_results, correct_score, crop = recognizer.recognize(TEST_IMAGE, expected_slug=CORRECT_SLUG)
    
    if crop is not None:
        cv2.imwrite("debug_crop_final.jpg", crop)
        
    if top_results is None:
        print("❌ YOLO не смог найти этикетку на фото.")
    else:
        print("🏆 ТОП-5 найденных совпадений в базе:")
        for i, res in enumerate(top_results):
            print(f"  {i+1}. {res.payload['slug']} (Сходство: {res.score:.3f})")
            
        print("\n🎯 Поиск правильного вина:")
        if correct_score is not None:
            print(f"  Slug: {CORRECT_SLUG}")
            print(f"  Точность: {correct_score:.3f}")
        else:
            print(f"  Слаг '{CORRECT_SLUG}' не найден в базе Qdrant!")

    recognizer.qdrant.close()