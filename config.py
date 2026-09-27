from contextvars import ContextVar
from functools import lru_cache

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class AppSettings(BaseSettings):

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        if not _settings_sources_enabled.get():
            return (init_settings,)

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    @classmethod
    def defaults(cls):
        token = _settings_sources_enabled.set(False)

        try:
            return cls()
        finally:
            _settings_sources_enabled.reset(token)


_settings_sources_enabled = ContextVar(
    "settings_sources_enabled",
    default=True,
)


class InferenceConfig(AppSettings):
    model_weights_path: str = Field(default="best_arcface_model.pth")
    yolo_weights_path: str = Field(default="best.pt")

    embedding_size: int = Field(default=512)
    image_size: int = Field(default=224)

    normalize_mean: tuple[float, float, float] = Field(default=(0.485, 0.456, 0.406))
    normalize_std: tuple[float, float, float] = Field(default=(0.229, 0.224, 0.225))

    device: str = Field(default="cuda:0")
    dataset_dir: str = Field(default="DATASTORE/dataset_arc/train")


class QdrantConfig(AppSettings):
    use_local_path: bool = Field(default=False)
    local_path: str = Field(default="qdrant_db")

    host: str = Field(default="qdrant")
    port: int = Field(default=6333)

    grpc_port: int = Field(default=6334)
    prefer_grpc: bool = Field(default=False)

    collection_name: str = Field(default="wines")


class RabbitMQConfig(AppSettings):
    host: str = Field(default="localhost")
    port: int = Field(default=5672)
    username: str = Field(default="guest")
    password: str = Field(default="guest")

    @property
    def url(self) -> str:
        return f"amqp://{self.username}:{self.password}@{self.host}:{self.port}/"

    publish_queue: str = Field(default="wine.recognition.requests")  # worker CONSUMES tasks from this queue
    consume_queue: str = Field(default="wine.recognition.results")  # worker PUBLISHES results to this queue

    prefetch_count: int = Field(default=1)


class MinioConfig(AppSettings):
    endpoint: str = Field(default="localhost:9000")

    access_key: str = Field(default="minioadmin")
    secret_key: str = Field(default="minioadmin")

    secure: bool = Field(default=False)

    image_bucket: str = Field(default="images")
    # image_prefix removed: object_key already contains the full prefix, so it's not needed for downloading from minio


class LoggingConfig(AppSettings):
    level: str = Field(default="INFO")
    logs_directory: str = Field(default="./logs/")
    file: str | None = Field(default=None)
    serialize: bool = Field(default=False)


class Config(AppSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        env_nested_delimiter="__",
        extra="ignore",
    )

    worker_name: str = Field(default="cv_inference_worker")
    worker_version: str = Field(default="0.0.1")

    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    rabbitmq: RabbitMQConfig = Field(default_factory=RabbitMQConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


@lru_cache
def config() -> Config:
    return Config()