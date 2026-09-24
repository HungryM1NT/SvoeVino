from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceConfig(BaseModel):
    model_weights_path: str = Field(default="best_arcface_model.pth")
    yolo_weights_path: str = Field(default="best.pt")

    embedding_size: int = Field(default=512)
    image_size: int = Field(default=224)

    normalize_mean: tuple[float, float, float] = Field(
        default=(0.485, 0.456, 0.406)
    )

    normalize_std: tuple[float, float, float] = Field(
        default=(0.229, 0.224, 0.225)
    )

    device: str = Field(default="cuda:0")
    dataset_dir: str = Field(default="DATASTORE/dataset_arc/train")


class QdrantConfig(BaseModel):
    use_local_path: bool = Field(default=False)
    local_path: str = Field(default="qdrant_db")

    host: str = Field(default="qdrant")
    port: int = Field(default=6333)

    grpc_port: int = Field(default=6334)
    prefer_grpc: bool = Field(default=False)

    collection_name: str = Field(default="wines")


class RabbitMQConfig(BaseModel):
    url: str = Field(
        default="amqp://guest:guest@localhost:5672/"
    )

    task_publish_queue: str = Field( # Worker READS this queue to get tasks
        default="wine.recognition.requests"
    )

    task_result_queue: str = Field( # Worker WRITES results to this queue
        default="wine.recognition.results"
    )

    prefetch_count: int = Field(default=1)


class MinioConfig(BaseModel):
    endpoint: str = Field(default="localhost:9000")

    access_key: str = Field(default="minioadmin")
    secret_key: str = Field(default="minioadmin")

    secure: bool = Field(default=False)

    image_bucket: str = Field(default="images")
    image_prefix: str = Field(default="client")


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        env_nested_delimiter="__",
        extra="ignore",
    )

    inference: InferenceConfig = Field(
        default_factory=InferenceConfig
    )

    qdrant: QdrantConfig = Field(
        default_factory=QdrantConfig
    )

    rabbitmq: RabbitMQConfig = Field(
        default_factory=RabbitMQConfig
    )

    minio: MinioConfig = Field(
        default_factory=MinioConfig
    )


@lru_cache
def config() -> Config:
    return Config()