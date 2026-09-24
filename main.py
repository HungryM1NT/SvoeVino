import asyncio
import json
import tempfile
from pathlib import Path

import aio_pika
from aio_pika import Message
from loguru import logger
from minio import Minio

from config import config
from inference import WineRecognizer
from qdrant_client import QdrantClient

def serialize_result(result) -> dict:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")

    if hasattr(result, "dict"):
        return result.dict()

    return {
        "id": getattr(result, "id", None),
        "score": getattr(result, "score", None),
        "payload": getattr(result, "payload", None),
    }


def normalize_object_key(
    object_key: str,
    prefix: str,
) -> str:
    object_key = object_key.lstrip("/")

    if not prefix:
        return object_key

    prefix = prefix.strip("/")

    if object_key == prefix or object_key.startswith(f"{prefix}/"):
        return object_key

    return f"{prefix}/{object_key}"


async def handle_message(
    message: aio_pika.abc.AbstractIncomingMessage,
    worker: WineRecognizer,
    minio_client: Minio,
    settings,
    channel: aio_pika.abc.AbstractChannel,
) -> None:
    async with message.process(requeue=False):
        try:
            payload = json.loads(message.body)

            request_id = payload["request_id"]
            object_key = payload["object_key"]
            expected_slug = payload.get("expected_slug")

        except (KeyError, json.JSONDecodeError) as exc:
            logger.error(f"Некорректное сообщение: {exc}")
            return

        object_key = normalize_object_key(
            object_key,
            settings.minio.image_prefix,
        )

        logger.info(
            f"Обработка запроса {request_id} "
            f"(object_key={object_key})"
        )

        try:
            suffix = Path(object_key).suffix or ".jpg"

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temp_file:
                image_path = temp_file.name

            try:
                await asyncio.to_thread(
                    minio_client.fget_object,
                    settings.minio.image_bucket,
                    object_key,
                    image_path,
                )

                top_results, expected_score, _ = await asyncio.to_thread(
                    worker.recognize,
                    image_path,
                    expected_slug,
                )

            finally:
                Path(image_path).unlink(
                    missing_ok=True
                )

            serialized_results = [
                serialize_result(result)
                for result in top_results
            ]

            response = {
                "request_id": request_id,
                "results": serialized_results,
                "expected_score": expected_score,
            }

        except Exception:
            logger.exception(
                f"Ошибка распознавания для запроса {request_id}"
            )

            response = {
                "request_id": request_id,
                "results": [],
                "expected_score": None,
                "error": "recognition_failed",
            }

        body = json.dumps(
            response,
            ensure_ascii=False,
        ).encode("utf-8")

        await channel.default_exchange.publish(
            Message(
                body=body,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=settings.rabbitmq.publish_queue,
        )

        logger.info(
            f"Ответ для запроса {request_id} отправлен"
        )


async def main() -> None:
    settings = config()

    logger.add(
        "logs/inference.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
        encoding="utf-8",
    )

    logger.info(
        "Загрузка модели распознавания..."
    )
    
    if not settings.qdrant.use_local_path:
        logger.info(
            f"Подключение к Qdrant: "
            f"{settings.qdrant.host}:{settings.qdrant.port}"
        )
        qdrant_client = QdrantClient(
            host=settings.qdrant.host,
            port=settings.qdrant.port,
            prefer_grpc=settings.qdrant.prefer_grpc,
            grpc_port=settings.qdrant.grpc_port,
        )
    else:
        logger.info(
            f"Использование локального пути для Qdrant: "
            f"{settings.qdrant.local_path}"
        )
        qdrant_client = QdrantClient(
            path=settings.qdrant.local_path,
        )

    worker = WineRecognizer(
        yolo_weights=settings.inference.yolo_weights_path,
        metric_weights=settings.inference.model_weights_path,
        qdrant=qdrant_client,
        collection_name=settings.qdrant.collection_name,
    )

    minio_client = Minio(
        settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )

    logger.info(
        "Подключение к RabbitMQ..."
    )

    connection = await aio_pika.connect_robust(
        settings.rabbitmq.url
    )

    async with connection:
        channel = await connection.channel()

        await channel.set_qos(
            prefetch_count=settings.rabbitmq.prefetch_count
        )

        consume_queue = await channel.declare_queue(
            settings.rabbitmq.consume_queue,
            durable=True,
        )

        await channel.declare_queue(
            settings.rabbitmq.publish_queue,
            durable=True,
        )

        logger.info(
            f"Ожидание сообщений в "
            f"'{settings.rabbitmq.consume_queue}'..."
        )

        async with consume_queue.iterator() as queue_iter:
            async for message in queue_iter:
                await handle_message(
                    message=message,
                    worker=worker,
                    minio_client=minio_client,
                    settings=settings,
                    channel=channel,
                )


if __name__ == "__main__":
    asyncio.run(main())