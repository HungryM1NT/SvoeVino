import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

# --- Параметры ---
DATASET_DIR = "dataset_prepared/train"
MODEL_WEIGHTS = "best_arcface_model.pth"
QDRANT_PATH = "qdrant_db"  # Папка, где будет лежать база векторов
COLLECTION_NAME = "wines"
EMBEDDING_SIZE = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 1. Облегченная модель (Только Backbone + Neck) ---
class InferenceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        # pretrained=False, так как мы загрузим свои веса
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.neck = nn.Sequential(
            nn.Linear(self.backbone.num_features, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.neck(features)
        return F.normalize(embeddings, p=2, dim=1) # Обязательная L2 нормализация

def main():
    print(f"Подготовка модели на {DEVICE}...")
    model = InferenceNet(EMBEDDING_SIZE).to(DEVICE)
    
    # Загружаем веса. strict=False позволяет проигнорировать веса arcface.weight из чекпойнта
    checkpoint = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # Трансформации (строго такие же, как при обучении)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 2. Инициализация Qdrant ---
    print(f"Инициализация базы Qdrant в '{QDRANT_PATH}'...")
    client = QdrantClient(path=QDRANT_PATH)
    
    # Создаем/пересоздаем коллекцию
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
    )

    # --- 3. Индексация ---
    slugs = [d.name for d in os.scandir(DATASET_DIR) if d.is_dir()]
    points = []
    
    print("Генерация векторов...")
    with torch.no_grad():
        for idx, slug in enumerate(tqdm(slugs)):
            # Берем идеальное (эталонное) изображение из папки
            ideal_img_path = os.path.join(DATASET_DIR, slug, "ideal.jpg")
            if not os.path.exists(ideal_img_path):
                continue
                
            img = Image.open(ideal_img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(DEVICE) # Добавляем размерность батча
            
            # Получаем вектор [1, 512] и переводим в обычный список float
            embedding = model(tensor).cpu().numpy()[0].tolist()
            
            # Формируем запись для базы данных
            point = PointStruct(
                id=idx, 
                vector=embedding, 
                payload={"slug": slug} # Метаданные, которые вернутся при поиске
            )
            points.append(point)
            
            # Загружаем пачками по 500 штук, чтобы не переполнять память
            if len(points) >= 500:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                points = []
                
    # Загружаем остатки
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Успешно проиндексировано вин: {len(slugs)}.")
    print("Система готова к поиску!")

if __name__ == "__main__":
    main()