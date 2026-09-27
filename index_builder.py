import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from loguru import logger
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from torchvision import transforms
from tqdm import tqdm

from config import Config, config


class InferenceNet(nn.Module):
    """Backbone + neck used to generate reference embeddings for the Qdrant index."""

    def __init__(self, embedding_size: int):
        super().__init__()
        # pretrained=False: we load our own fine-tuned weights below.
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.neck = nn.Sequential(
            nn.Linear(self.backbone.num_features, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.neck(features)
        return F.normalize(embeddings, p=2, dim=1)  # mandatory L2 normalization


def build_qdrant_client(settings: Config) -> QdrantClient:
    """Shared with the worker so index building and inference always talk to the same Qdrant."""
    if settings.qdrant.use_local_path:
        logger.info("Using local Qdrant path: {}", settings.qdrant.local_path)
        return QdrantClient(path=settings.qdrant.local_path)

    logger.info("Connecting to Qdrant: {}:{}", settings.qdrant.host, settings.qdrant.port)
    return QdrantClient(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        prefer_grpc=settings.qdrant.prefer_grpc,
        grpc_port=settings.qdrant.grpc_port,
    )


def _load_model(settings: Config, device: torch.device) -> InferenceNet:
    model = InferenceNet(settings.inference.embedding_size).to(device)

    checkpoint = torch.load(settings.inference.model_weights_path, map_location=device)
    # strict=False ignores the arcface.weight head left over from the training checkpoint
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    return model


def _build_transform(settings: Config) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((settings.inference.image_size, settings.inference.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(settings.inference.normalize_mean),
                std=list(settings.inference.normalize_std),
            ),
        ]
    )


def build_index(
    settings: Config,
    dataset_dir: str | None = None,
    batch_size: int = 500,
) -> None:
    device = torch.device(settings.inference.device if torch.cuda.is_available() else "cpu")
    dataset_path = Path(dataset_dir or settings.inference.dataset_dir)

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    logger.info("Preparing model on {}", device)
    model = _load_model(settings, device)
    transform = _build_transform(settings)

    client = build_qdrant_client(settings)

    logger.info("(Re)creating collection '{}'", settings.qdrant.collection_name)
    client.recreate_collection(
        collection_name=settings.qdrant.collection_name,
        vectors_config=VectorParams(
            size=settings.inference.embedding_size,
            distance=Distance.COSINE,
        ),
    )

    slugs = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    points = []

    logger.info("Generating embeddings for {} wines from {}", len(slugs), dataset_path)

    with torch.no_grad():
        for idx, slug in enumerate(tqdm(slugs)):
            ideal_image_path = dataset_path / slug / "ideal.jpg"
            if not ideal_image_path.exists():
                logger.warning("Skipping '{}': no ideal.jpg found", slug)
                continue

            image = Image.open(ideal_image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            embedding = model(tensor).cpu().numpy()[0].tolist()

            points.append(
                PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={"slug": slug},
                )
            )

            if len(points) >= batch_size:
                client.upsert(collection_name=settings.qdrant.collection_name, points=points)
                points = []

    if points:
        client.upsert(collection_name=settings.qdrant.collection_name, points=points)

    logger.info("Indexed {} wines into '{}'", len(slugs), settings.qdrant.collection_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Qdrant reference index for wine recognition.")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Override the reference dataset directory (defaults to inference.dataset_dir from config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of points to upsert per Qdrant batch.",
    )
    args = parser.parse_args()

    settings = config()
    build_index(settings, dataset_dir=args.dataset_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()