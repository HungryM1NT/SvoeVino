import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

# --- Параметры обучения ---
DATASET_DIR = "dataset_prepared"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

EMBEDDING_SIZE = 512       # Размерность итогового вектора
BATCH_SIZE = 128           # Оптимально для 16 ГБ VRAM при 224x224
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- 1. Слой ArcFace Loss ---
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, label):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# --- 2. Модель EfficientNet + Neck + ArcFace ---
class WineEmbeddingNet(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super().__init__()
        # Предобученный бэкбоун EfficientNet-B0
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        in_features = self.backbone.num_features

        # Проекционный слой с Batch Normalization
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size)
        )
        self.arcface = ArcMarginProduct(embedding_size, num_classes)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        embeddings = self.neck(features)
        # Обязательная L2-нормализация для сферического пространства
        embeddings = F.normalize(embeddings, p=2, dim=1)

        if labels is not None:
            logits = self.arcface(embeddings, labels)
            return logits, embeddings
        return embeddings


def main():
    print(f"Используемое устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # --- 3. Подготовка трансформаций и DataLoader ---
    # Поскольку аугментации уже применены на диске, здесь только ресайз и нормализация ImageNet
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    num_classes = len(train_dataset.classes)
    print(f"Обнаружено классов (вин): {num_classes}")
    print(f"Обучающих изображений: {len(train_dataset)} | Валидационных: {len(val_dataset)}")

    # Сохраняем маппинг ID класса -> slug
    idx_to_slug = {idx: slug for slug, idx in train_dataset.class_to_idx.items()}
    with open("slugs_map.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_slug, f, ensure_ascii=False, indent=2)
    print("Карта классов сохранена в slugs_map.json")

    # --- 4. Инициализация обучения ---
    model = WineEmbeddingNet(num_classes=num_classes, embedding_size=EMBEDDING_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0

    # --- 5. Тренировочный цикл ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss, train_correct, train_total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                logits, _ = model(images, labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            train_bar.set_postfix(loss=loss.item(), acc=f"{100.0 * train_correct / train_total:.2f}%")

        scheduler.step()
        train_acc = 100.0 * train_correct / train_total

        # --- 6. Валидация ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    logits, _ = model(images, labels)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        print(f"Результаты Эпохи {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss / val_total:.4f}")

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'embedding_size': EMBEDDING_SIZE,
                'val_acc': val_acc
            }, "best_arcface_model.pth")
            print(f">> Сохранены новые лучшие веса (Val Acc: {val_acc:.2f}%) в best_arcface_model.pth")

    print("\nОбучение завершено!")


if __name__ == "__main__":
    main()