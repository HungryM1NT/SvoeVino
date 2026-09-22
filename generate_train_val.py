import os
import random
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# --- Параметры пайплайна ---
SOURCE_DIR = Path("dataset/train")        # Папка, где сейчас лежат папки со слагами и 1 фото
OUTPUT_DIR = Path("dataset_prepared")     # Куда разложить готовые train и val
TARGET_SIZE = 640                         # Размер квадрата на выходе
BG_COLOR = (128, 128, 128)               # Серый фон (B, G, R)

TRAIN_VARIANTS = 35                       # Количество синтетических копий в train
VAL_VARIANTS = 10                         # Количество синтетических копий в val


class FastBottleSynthesizer:
    def __init__(self, target_size=640, bg_color=(128, 128, 128)):
        self.target_size = target_size
        self.bg_color = np.array(bg_color, dtype=np.uint8)

    def synthesize_variant(self, img_bgra):
        h, w = img_bgra.shape[:2]

        # 1. Случайные параметры геометрии и камеры
        fov_deg = random.uniform(115, 155)
        theta_max = np.deg2rad(fov_deg / 2.0)
        R = (w / 2.0) / theta_max

        pitch_amplitude = random.uniform(-0.12, 0.12) * h
        yaw_offset = random.uniform(-0.35, 0.35) * theta_max

        # Освещение
        light_x = random.uniform(-1.0, 1.0)
        shininess = random.uniform(35.0, 85.0)
        glare_intensity = random.uniform(100.0, 230.0)

        # 2. Векторизованная генерация 3D-сетки цилиндра
        w_out = int(2 * R * np.sin(theta_max))
        h_out = h + int(abs(pitch_amplitude))

        x_indices = np.arange(w_out, dtype=np.float32)
        val = np.clip((x_indices - w_out / 2.0) / R, -1.0, 1.0)
        theta = np.arcsin(val)

        theta_src = theta + yaw_offset
        valid_mask_1d = np.abs(theta_src) <= theta_max

        # Координаты X для ремапа
        x_src_1d = np.where(
            valid_mask_1d,
            (theta_src / theta_max) * (w / 2.0) + (w / 2.0),
            -1.0
        )
        map_x = np.tile(x_src_1d, (h_out, 1)).astype(np.float32)
        alpha_mask = np.tile(valid_mask_1d.astype(np.float32), (h_out, 1))

        # Нормали поверхности
        normals_x = np.tile(np.sin(theta), (h_out, 1)).astype(np.float32)
        normals_z = np.tile(np.cos(theta), (h_out, 1)).astype(np.float32)

        # Координаты Y с учетом наклона (Pitch)
        y_curve_offset = pitch_amplitude * (np.cos(theta) - 1.0)
        y_indices = np.arange(h_out, dtype=np.float32)[:, np.newaxis]
        map_y = (y_indices - max(0, pitch_amplitude) - y_curve_offset).astype(np.float32)

        # 3. 3D Warp
        warped = cv2.remap(
            img_bgra, map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0, 0)
        )

        bgr = warped[:, :, :3].astype(np.float32)
        alpha = (warped[:, :, 3].astype(np.float32) / 255.0) * alpha_mask

        # 4. Физическая модель освещения (Blinn-Phong)
        light_dir = np.array([light_x, 0.2, 0.8], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        half_vec = light_dir + view_dir
        half_vec /= np.linalg.norm(half_vec)

        diffuse = np.clip(normals_x * light_dir[0] + normals_z * light_dir[2], 0.0, 1.0)
        ambient = 0.45
        shading = ambient + (1.0 - ambient) * diffuse

        spec_dot = np.clip(normals_x * half_vec[0] + normals_z * half_vec[2], 0.0, 1.0)
        specular = np.power(spec_dot, shininess) * glare_intensity

        shading_3d = np.stack([shading] * 3, axis=2)
        specular_3d = np.stack([specular] * 3, axis=2)

        final_bgr = np.clip((bgr * shading_3d) + specular_3d, 0.0, 255.0)

        # 5. Шум сенсора
        noise = np.random.normal(0, random.uniform(2.0, 5.0), final_bgr.shape).astype(np.float32)
        final_bgr = np.clip(final_bgr + noise, 0.0, 255.0)

        # 6. Композиция на серый фон вместо прозрачности
        mask_3d = np.stack([alpha], axis=2)
        bg = np.full_like(final_bgr, self.bg_color, dtype=np.float32)
        blended = (final_bgr * mask_3d + bg * (1.0 - mask_3d)).astype(np.uint8)

        # 7. Letterboxing в целевой квадрат
        return self._letterbox(blended)

    def _letterbox(self, img):
        h, w = img.shape[:2]
        scale = self.target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.target_size, self.target_size, 3), self.bg_color, dtype=np.uint8)

        x_off = (self.target_size - new_w) // 2
        y_off = (self.target_size - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas


def main():
    synthesizer = FastBottleSynthesizer(target_size=TARGET_SIZE, bg_color=BG_COLOR)

    # Ищем все папки со слагами
    slug_dirs = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    print(f"Найдено папок вин: {len(slug_dirs)}")

    for slug_dir in tqdm(slug_dirs, desc="Обработка вин"):
        slug = slug_dir.name

        # Находим исходное изображение
        images = list(slug_dir.glob("*.jpg")) + list(slug_dir.glob("*.webp")) + list(slug_dir.glob("*.png"))
        if not images:
            continue

        ideal_img_path = images[0]
        img = cv2.imread(str(ideal_img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # Приводим к 4 каналам (BGRA) для корректного маскирования
        if img.ndim == 2:
            img_bgra = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        else:
            img_bgra = img

        train_slug_dir = OUTPUT_DIR / "train" / slug
        val_slug_dir = OUTPUT_DIR / "val" / slug
        train_slug_dir.mkdir(parents=True, exist_ok=True)
        val_slug_dir.mkdir(parents=True, exist_ok=True)

        # 1. В train кладем оригинал (с серым паддингом)
        base_canvas = synthesizer._letterbox(cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR))
        cv2.imwrite(str(train_slug_dir / "ideal.jpg"), base_canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 2. Генерируем тренировочные искажения (TRAIN)
        for i in range(TRAIN_VARIANTS):
            synth_img = synthesizer.synthesize_variant(img_bgra)
            out_file = train_slug_dir / f"synth_{i:02d}.jpg"
            cv2.imwrite(str(out_file), synth_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # 3. Генерируем независимые валидационные искажения (VAL)
        for i in range(VAL_VARIANTS):
            synth_img = synthesizer.synthesize_variant(img_bgra)
            out_file = val_slug_dir / f"val_synth_{i:02d}.jpg"
            cv2.imwrite(str(out_file), synth_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print("\nГенерация завершена успешно!")
    print(f"Результаты сохранены в: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()