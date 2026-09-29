import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image

from config import Config, config


TARGET_SIZE = 640
BG_COLOR = (128, 128, 128)
YOLO_MODEL_PATH = "models/YOLO_best_100.pt"


class InferenceNet(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0,
        )
        self.neck = nn.Sequential(
            nn.Linear(self.backbone.num_features, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.neck(features)
        return F.normalize(embeddings, p=2, dim=1)


def build_qdrant_client(settings: Config) -> QdrantClient:
    if settings.qdrant.use_local_path:
        logger.info("Using local Qdrant path: {}", settings.qdrant.local_path)
        return QdrantClient(path=settings.qdrant.local_path)

    logger.info(
        "Connecting to Qdrant: {}:{}",
        settings.qdrant.host,
        settings.qdrant.port,
    )

    return QdrantClient(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        prefer_grpc=settings.qdrant.prefer_grpc,
        grpc_port=settings.qdrant.grpc_port,
    )


def _load_model(settings: Config, device: torch.device) -> InferenceNet:
    model = InferenceNet(settings.inference.embedding_size).to(device)

    checkpoint = torch.load(
        settings.inference.model_weights_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    model.eval()

    return model


def _build_transform(settings: Config) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (
                    settings.inference.image_size,
                    settings.inference.image_size,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(settings.inference.normalize_mean),
                std=list(settings.inference.normalize_std),
            ),
        ]
    )


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    cl = clahe.apply(l)

    return cv2.cvtColor(
        cv2.merge((cl, a, b)),
        cv2.COLOR_LAB2BGR,
    )


def smart_letterbox(
    img: np.ndarray,
    target_size: int,
    bg_color: tuple[int, int, int],
) -> np.ndarray:
    h, w = img.shape[:2]

    scale = target_size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=interp,
    )

    canvas = np.full(
        (target_size, target_size, 3),
        bg_color,
        dtype=np.uint8,
    )

    x_off = (target_size - new_w) // 2
    y_off = (target_size - new_h) // 2

    canvas[
        y_off:y_off + new_h,
        x_off:x_off + new_w,
    ] = resized

    return canvas


def extract_label_yolo(
    img: np.ndarray,
    yolo_model: YOLO,
) -> np.ndarray:
    results = yolo_model.predict(
        source=img,
        conf=0.5,
        retina_masks=True,
        verbose=False,
    )

    if (
        not results
        or len(results[0].boxes) == 0
        or results[0].masks is None
    ):
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    result = results[0]

    best_idx = np.argmax(
        result.boxes.conf.cpu().numpy()
    )

    box = (
        result.boxes.xyxy[best_idx]
        .cpu()
        .numpy()
        .astype(int)
    )

    polygon = result.masks.xy[best_idx].astype(
        np.int32
    )

    img_bgra = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2BGRA,
    )

    blank_mask = np.zeros(
        img.shape[:2],
        dtype=np.uint8,
    )

    cv2.fillPoly(
        blank_mask,
        [polygon],
        255,
    )

    img_bgra[:, :, 3] = blank_mask

    x1, y1, x2, y2 = box

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    return img_bgra[y1:y2, x1:x2]


def preprocess_reference_image(
    image_path: str,
    yolo_model: YOLO,
) -> Image.Image:
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(
            f"Failed to read image: {image_path}"
        )

    img_bgra = extract_label_yolo(
        img,
        yolo_model,
    )

    bgr_clahe = apply_clahe(
        img_bgra[:, :, :3]
    )

    alpha = (
        img_bgra[:, :, 3].astype(np.float32)
        / 255.0
    )[:, :, np.newaxis]

    gray_bg = np.full_like(
        bgr_clahe,
        BG_COLOR,
        dtype=np.uint8,
    )

    flattened_ideal = (
        bgr_clahe * alpha
        + gray_bg * (1.0 - alpha)
    ).astype(np.uint8)

    ideal_canvas = smart_letterbox(
        flattened_ideal,
        TARGET_SIZE,
        BG_COLOR,
    )

    rgb = cv2.cvtColor(
        ideal_canvas,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(rgb)


def find_image(
    input_dir: Path,
    scraped_id: str,
) -> str | None:
    pattern = os.path.join(
        str(input_dir),
        f"wine_{scraped_id}_*.*",
    )

    found_files = glob.glob(pattern)

    if not found_files:
        found_files = glob.glob(
            os.path.join(
                str(input_dir),
                f"wine_{scraped_id}.*",
            )
        )

    if not found_files:
        return None

    return found_files[0]


def build_index(
    settings: Config,
    dataset_dir: str | None = None,
    batch_size: int = 500,
    yolo_model_path: str = YOLO_MODEL_PATH,
    qdrant_use_remote: bool = False,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    qdrant_grpc_port: int | None = None,
    qdrant_prefer_grpc: bool | None = None,
) -> None:
    dataset_path = Path(
        dataset_dir or settings.inference.dataset_dir
    )

    wine_png_dir = dataset_path / "wine_png"
    metadata_path = dataset_path / "wine_metadata.json"

    if not wine_png_dir.is_dir():
        raise FileNotFoundError(
            f"wine_png directory not found: {wine_png_dir}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"wine_metadata.json not found: {metadata_path}"
        )

    device = torch.device(
        settings.inference.device
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info("Using device: {}", device)

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    logger.info(
        "Loaded {} metadata records",
        len(metadata),
    )

    logger.info(
        "Loading YOLO model from {}",
        yolo_model_path,
    )

    yolo_model = YOLO(yolo_model_path)

    logger.info("Preparing embedding model")

    model = _load_model(
        settings,
        device,
    )

    transform = _build_transform(
        settings
    )

    if qdrant_use_remote:
        settings.qdrant.use_local_path = False

        if qdrant_host is not None:
            settings.qdrant.host = qdrant_host

        if qdrant_port is not None:
            settings.qdrant.port = qdrant_port

        if qdrant_grpc_port is not None:
            settings.qdrant.grpc_port = (
                qdrant_grpc_port
            )

        if qdrant_prefer_grpc is not None:
            settings.qdrant.prefer_grpc = (
                qdrant_prefer_grpc
            )

    logger.info(
        "Qdrant: local={} host={} port={} grpc={}",
        settings.qdrant.use_local_path,
        settings.qdrant.host,
        settings.qdrant.port,
        settings.qdrant.grpc_port,
    )

    client = build_qdrant_client(
        settings
    )

    logger.info(
        "(Re)creating collection '{}'",
        settings.qdrant.collection_name,
    )

    client.recreate_collection(
        collection_name=settings.qdrant.collection_name,
        vectors_config=VectorParams(
            size=settings.inference.embedding_size,
            distance=Distance.COSINE,
        ),
    )

    points: list[PointStruct] = []

    indexed_count = 0
    skipped_count = 0

    logger.info(
        "Generating embeddings from raw dataset: {}",
        dataset_path,
    )

    with torch.no_grad():
        for idx, item in enumerate(
            tqdm(
                metadata,
                desc="Building Qdrant index",
            )
        ):
            scraped_id = item.get(
                "scraped_id"
            )

            slug = (
                item.get("raw_metadata", {})
                .get("slug")
            )

            if not scraped_id or not slug:
                logger.warning(
                    "Skipping metadata record {}: missing scraped_id or slug",
                    idx,
                )
                skipped_count += 1
                continue

            image_path = find_image(
                wine_png_dir,
                str(scraped_id),
            )

            if image_path is None:
                logger.warning(
                    "Skipping '{}': source image not found for scraped_id={}",
                    slug,
                    scraped_id,
                )
                skipped_count += 1
                continue

            try:
                image = preprocess_reference_image(
                    image_path,
                    yolo_model,
                )

                tensor = (
                    transform(image)
                    .unsqueeze(0)
                    .to(device)
                )

                embedding = (
                    model(tensor)
                    .cpu()
                    .numpy()[0]
                    .tolist()
                )

            except Exception as exc:
                logger.exception(
                    "Failed to process '{}' from '{}': {}",
                    slug,
                    image_path,
                    exc,
                )
                skipped_count += 1
                continue

            points.append(
                PointStruct(
                    id=indexed_count,
                    vector=embedding,
                    payload={
                        "slug": slug,
                        "scraped_id": scraped_id,
                    },
                )
            )

            indexed_count += 1

            if len(points) >= batch_size:
                client.upsert(
                    collection_name=settings.qdrant.collection_name,
                    points=points,
                )
                points = []

    if points:
        client.upsert(
            collection_name=settings.qdrant.collection_name,
            points=points,
        )

    logger.info(
        "Indexed {} wines into '{}'",
        indexed_count,
        settings.qdrant.collection_name,
    )

    logger.info(
        "Skipped {} wines",
        skipped_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Qdrant reference index "
            "directly from the raw wine dataset."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "Dataset root containing wine_png/ "
            "and wine_metadata.json."
        ),
    )

    parser.add_argument(
        "--yolo-model",
        default=YOLO_MODEL_PATH,
        help="Path to YOLO segmentation weights.",
    )

    parser.add_argument(
        "--qdrant-dir",
        default=None,
        help=(
            "Override local Qdrant path."
        ),
    )

    parser.add_argument(
        "--qdrant-use-remote",
        action="store_true",
        help="Use remote Qdrant.",
    )

    parser.add_argument(
        "--qdrant-host",
        default=None,
        help="Override Qdrant host.",
    )

    parser.add_argument(
        "--qdrant-port",
        default=None,
        type=int,
        help="Override Qdrant HTTP port.",
    )

    parser.add_argument(
        "--qdrant-grpc-port",
        default=None,
        type=int,
        help="Override Qdrant gRPC port.",
    )

    parser.add_argument(
        "--qdrant-prefer-grpc",
        action="store_true",
        help="Use gRPC for Qdrant.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of points per Qdrant upsert.",
    )

    args = parser.parse_args()

    settings = config()

    if args.qdrant_dir is not None:
        settings.qdrant.local_path = args.qdrant_dir

    build_index(
        settings=settings,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        yolo_model_path=args.yolo_model,
        qdrant_use_remote=args.qdrant_use_remote,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_grpc_port=args.qdrant_grpc_port,
        qdrant_prefer_grpc=(
            args.qdrant_prefer_grpc
            if args.qdrant_prefer_grpc
            else None
        ),
    )


if __name__ == "__main__":
    main()

