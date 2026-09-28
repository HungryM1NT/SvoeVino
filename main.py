import argparse
import asyncio
import json
import signal
import sys
import tempfile
from pathlib import Path

import aio_pika
from aio_pika import DeliveryMode, Message
from config import Config, config
from index_builder import build_index, build_qdrant_client
from loguru import logger
from minio import Minio
from pydantic import BaseModel, ValidationError

from inference import WineRecognizer


class TaskMessage(BaseModel):
    """Contract for incoming task messages."""

    task_id: str
    object_key: str


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


class WineRecognitionWorker:
    def __init__(self, settings: Config):
        self.settings = settings

        self.worker: WineRecognizer | None = None
        self.minio_client: Minio | None = None
        self.connection: aio_pika.abc.AbstractRobustConnection | None = None
        self.channel: aio_pika.abc.AbstractChannel | None = None
        self.task_queue: aio_pika.abc.AbstractQueue | None = None

        self._stop_event = asyncio.Event()

    def logger_setup(self) -> None:
        logger.remove()

        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:"
            "<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        logger.add(
            sys.stderr,
            level=self.settings.logging.level,
            format=log_format,
            serialize=self.settings.logging.serialize,
            backtrace=True,
            diagnose=False,
        )

        if self.settings.logging.file:
            log_path = Path(self.settings.logging.logs_directory) / self.settings.logging.file
            log_path.parent.mkdir(parents=True, exist_ok=True)

            logger.add(
                log_path,
                level=self.settings.logging.level,
                format=log_format,
                serialize=self.settings.logging.serialize,
                backtrace=True,
                diagnose=False,
                encoding="utf-8",
            )

        logger.info(
            "Logger setup complete: worker={} version={} level={}",
            self.settings.worker_name,
            self.settings.worker_version,
            self.settings.logging.level,
        )

    async def prelude(self) -> None:
        logger.info("Starting worker prelude routine")

        logger.debug("Initializing Qdrant client")
        qdrant_client = build_qdrant_client(self.settings)

        logger.debug("Loading wine recognizer")
        self.worker = WineRecognizer(
            yolo_weights=self.settings.inference.yolo_weights_path,
            metric_weights=self.settings.inference.model_weights_path,
            qdrant=qdrant_client,
            collection_name=self.settings.qdrant.collection_name,
        )
        logger.info(
            "Recognizer initialized: yolo={}, metric={}, collection={}",
            self.settings.inference.yolo_weights_path,
            self.settings.inference.model_weights_path,
            self.settings.qdrant.collection_name,
        )

        logger.debug("Creating Minio client")
        self.minio_client = Minio(
            self.settings.minio.endpoint,
            access_key=self.settings.minio.access_key,
            secret_key=self.settings.minio.secret_key,
            secure=self.settings.minio.secure,
        )

        logger.debug("Connecting to RabbitMQ")
        try:
            self.connection = await aio_pika.connect_robust(self.settings.rabbitmq.url)
        except Exception:
            logger.exception("Failed to connect to RabbitMQ")
            raise
        logger.debug("RabbitMQ connection established")

        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=self.settings.rabbitmq.prefetch_count)

        # publish_queue: backend publishes tasks here -> worker consumes from it
        self.task_queue = await self.channel.declare_queue(
            self.settings.rabbitmq.publish_queue,
            durable=True,
        )
        # consume_queue: backend consumes results from here -> worker publishes to it
        await self.channel.declare_queue(
            self.settings.rabbitmq.consume_queue,
            durable=True,
        )

        logger.info("Worker prelude routine completed successfully")

    async def publish_result(self, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        await self.channel.default_exchange.publish(
            Message(
                body=body,
                content_type="application/json",
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key=self.settings.rabbitmq.consume_queue,
        )

    @staticmethod
    def _error_response(request_id: str | None, error: str) -> dict:
        return {
            "request_id": request_id,
            "results": [],
            "expected_score": None,
            "error": error,
        }

    async def handle_message(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        async with message.process(requeue=False):
            try:
                task = TaskMessage.model_validate_json(message.body)
            except (ValidationError, ValueError) as exc:
                # Covers malformed JSON and contract violations (missing/extra/wrong-typed fields)
                logger.error("Rejected malformed task message: {}", exc)
                return

            logger.info(
                "Received request {}: object_key={}",
                task.task_id,
                task.object_key,
            )

            image_path: str | None = None

            try:
                suffix = Path(task.object_key).suffix or ".jpg"

                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                    image_path = temp_file.name

                await asyncio.to_thread(
                    self.minio_client.fget_object,
                    self.settings.minio.image_bucket,
                    task.object_key,
                    image_path,
                )

                top_results, _, _ = await asyncio.to_thread(
                    self.worker.recognize,
                    image_path,
                )

                response = {
                    "task_id": task.task_id,
                    "result": [serialize_result(result) for result in top_results],
                }

                logger.info("Request {} finished successfully", task.task_id)

            except Exception:
                logger.exception("Recognition failed for request {}", task.task_id)
                response = self._error_response(task.task_id, "recognition_failed")

            finally:
                if image_path is not None:
                    Path(image_path).unlink(missing_ok=True)

            await self.publish_result(response)
            logger.info(
                "Result for request {} published to '{}'",
                task.task_id,
                self.settings.rabbitmq.consume_queue,
            )

    async def consume(self) -> None:
        async with self.task_queue.iterator() as queue_iter:
            async for message in queue_iter:
                if self._stop_event.is_set():
                    break
                await self.handle_message(message)

    async def shutdown(self) -> None:
        logger.info("Shutting down worker")
        self._stop_event.set()
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
        logger.info("Worker shutdown complete")

    async def run(self) -> None:
        self.logger_setup()
        await self.prelude()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                # add_signal_handler is unavailable on some platforms (e.g. Windows)
                pass

        logger.info("Waiting for tasks in '{}'...", self.settings.rabbitmq.publish_queue)

        try:
            await self.consume()
        finally:
            if self.connection and not self.connection.is_closed:
                await self.connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Wine CV Recognition Worker")
    parser.add_argument(
        "--generate-config-json-string",
        action="store_true",
        help="Generate a JSON string of the default config and exit.",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build the Qdrant reference index from the dataset and exit.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Override the reference dataset directory used with --build-index.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Qdrant upsert batch size used with --build-index.",
    )

    args = parser.parse_args()

    if args.generate_config_json_string:
        print(Config.defaults().model_dump_json())
        return

    if args.build_index:
        settings = config()
        build_index(settings, dataset_dir=args.dataset_dir, batch_size=args.batch_size)
        return

    settings = config()
    worker = WineRecognitionWorker(settings)
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()