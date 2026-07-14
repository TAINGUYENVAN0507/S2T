import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable
from inspect import isawaitable
from pathlib import Path
from typing import Any, TypeVar, cast

from fastapi import HTTPException, UploadFile, status
from redis.asyncio import Redis as AsyncRedis

from app.config import settings
from app.schemas.response import (
    AudioBatchCreatedResponse,
    AudioBatchStatusResponse,
    AudioFileResponse,
    AudioStatus,
    BatchStatus,
    QueueStatusResponse,
)
from app.workers.celery import process_audio_task

logger = logging.getLogger(__name__)

redis_client: AsyncRedis = AsyncRedis.from_url(
    settings.REDIS_URL,
)

RedisValue = bytes | str | int | float
T = TypeVar("T")

VALID_AUDIO_STATUSES: set[AudioStatus] = {
    "queued",
    "processing",
    "success",
    "failed",
    "cancelled",
}
TERMINAL_BATCH_STATUSES: set[BatchStatus] = {
    "completed",
    "partial_failed",
    "failed",
    "cancelled",
}


ALLOWED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".m4a",
    ".wav",
    ".flac",
    ".ogg",
    ".webm",
    ".mp4",
    ".mpeg",
    ".mpga",
}


def build_audio_key(batch_id: str, file_id: str) -> str:
    return f"audio:{batch_id}:{file_id}:source"


def build_file_key(batch_id: str, file_id: str) -> str:
    return f"file:{batch_id}:{file_id}"


def build_batch_key(batch_id: str) -> str:
    return f"batch:{batch_id}"


def build_batch_files_key(batch_id: str) -> str:
    return f"batch:{batch_id}:files"


async def hset_mapping(
    key: str,
    mapping: dict[str, Any],
) -> int:
    return int(
        await resolve_redis(
            redis_client.hset(
                key,
                mapping={field: str(value) for field, value in mapping.items()},
            )
        )
    )


async def resolve_redis(result: T | Awaitable[T]) -> T:
    if isawaitable(result):
        return await cast(Awaitable[T], result)

    return result


async def redis_get(key: str) -> RedisValue | None:
    return await resolve_redis(redis_client.get(key))


async def redis_set(
    key: str,
    value: RedisValue,
    *,
    ex: int | None = None,
) -> Any:
    return await resolve_redis(
        redis_client.set(
            key,
            value,
            ex=ex,
        )
    )


async def redis_expire(
    key: str,
    seconds: int,
) -> bool:
    return bool(await resolve_redis(redis_client.expire(key, seconds)))


async def redis_incr(key: str) -> int:
    return int(await resolve_redis(redis_client.incr(key)))


async def redis_rpush(
    key: str,
    *values: RedisValue,
) -> int:
    return int(await resolve_redis(redis_client.rpush(key, *values)))


async def redis_hgetall(key: str) -> dict[Any, Any]:
    return await resolve_redis(redis_client.hgetall(key))


async def redis_lrange(
    key: str,
    start: int,
    end: int,
) -> list[Any]:
    return await resolve_redis(redis_client.lrange(key, start, end))


async def redis_exists(key: str) -> bool:
    return bool(await resolve_redis(redis_client.exists(key)))


def decode_redis_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)


def decode_redis_hash(data: dict[Any, Any]) -> dict[str, str]:
    return {
        decode_redis_value(key): decode_redis_value(value)
        for key, value in data.items()
    }


def normalize_audio_status(value: str | None) -> AudioStatus:
    if value in VALID_AUDIO_STATUSES:
        return value

    return "failed"


async def create_audio_batch(
    *,
    files: list[UploadFile],
    language: str,
    use_vad: bool,
) -> AudioBatchCreatedResponse:
    """
    Tạo batch xử lý audio.

    Luồng:
    1. Validate số lượng file.
    2. Kiểm tra Redis queue.
    3. Lưu audio bytes vào Redis.
    4. Lưu batch/file status vào Redis.
    5. Enqueue Celery task.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one audio file is required.",
        )

    if len(files) > settings.MAX_AUDIO_FILES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(f"Maximum number of audio files is {settings.MAX_AUDIO_FILES}."),
        )

    await ensure_queue_has_capacity(len(files))

    batch_id = f"batch_{uuid.uuid4().hex}"
    batch_key = build_batch_key(batch_id)
    batch_files_key = build_batch_files_key(batch_id)

    await hset_mapping(
        batch_key,
        {
            "batch_id": batch_id,
            "status": "queued",
            "total_files": len(files),
            "language": language,
            "use_vad": int(use_vad),
        },
    )

    await redis_expire(
        batch_key,
        settings.RESULT_TTL_SECONDS,
    )

    response_files: list[AudioFileResponse] = []

    for file in files:
        try:
            audio_file = await prepare_audio_file(
                batch_id=batch_id,
                file=file,
            )

            await enqueue_audio_task(
                batch_id=batch_id,
                file_id=audio_file.file_id,
                filename=audio_file.filename,
                audio_key=audio_file.audio_key,
                language=language,
                use_vad=use_vad,
            )

            response_files.append(
                AudioFileResponse(
                    file_id=audio_file.file_id,
                    filename=audio_file.filename,
                    status="queued",
                )
            )

        finally:
            await file.close()

    await redis_expire(
        batch_files_key,
        settings.RESULT_TTL_SECONDS,
    )

    return AudioBatchCreatedResponse(
        batch_id=batch_id,
        status="queued",
        total_files=len(response_files),
        files=response_files,
    )


class PreparedAudioFile:
    """
    Object nội bộ dùng sau khi validate và lưu audio vào Redis.
    """

    def __init__(
        self,
        *,
        file_id: str,
        filename: str,
        audio_key: str,
    ) -> None:
        self.file_id = file_id
        self.filename = filename
        self.audio_key = audio_key


async def prepare_audio_file(
    *,
    batch_id: str,
    file: UploadFile,
) -> PreparedAudioFile:
    """
    Validate file, đọc bytes, lưu audio gốc vào Redis.
    """
    validate_audio_file(file)

    file_id = f"file_{uuid.uuid4().hex}"
    filename = file.filename or file_id

    audio_bytes = await read_audio_bytes(file)

    audio_key = build_audio_key(
        batch_id=batch_id,
        file_id=file_id,
    )

    file_key = build_file_key(
        batch_id=batch_id,
        file_id=file_id,
    )

    batch_files_key = build_batch_files_key(batch_id)

    await redis_set(
        audio_key,
        audio_bytes,
        ex=settings.AUDIO_TTL_SECONDS,
    )

    await hset_mapping(
        file_key,
        {
            "file_id": file_id,
            "filename": filename,
            "status": "queued",
            "text": "",
            "error": "",
            "audio_key": audio_key,
        },
    )

    await redis_expire(
        file_key,
        settings.RESULT_TTL_SECONDS,
    )

    await redis_rpush(
        batch_files_key,
        file_id,
    )

    await redis_incr("pipeline:queued")

    return PreparedAudioFile(
        file_id=file_id,
        filename=filename,
        audio_key=audio_key,
    )


async def enqueue_audio_task(
    *,
    batch_id: str,
    file_id: str,
    filename: str,
    audio_key: str,
    language: str,
    use_vad: bool,
) -> None:
    """
    Đẩy task vào Celery.

    Task chỉ nhận key, không nhận audio bytes.
    """
    process_audio_task.apply_async(
        kwargs={
            "batch_id": batch_id,
            "file_id": file_id,
            "audio_key": audio_key,
            "filename": filename,
            "language": language,
            "use_vad": use_vad,
        },
        queue="audio_pipeline",
    )


def validate_audio_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio filename is missing.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio format: {extension}",
        )


async def read_audio_bytes(file: UploadFile) -> bytes:
    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio file is empty: {file.filename}",
        )

    max_size_bytes = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024

    if len(audio_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Audio file {file.filename} exceeds {settings.MAX_AUDIO_SIZE_MB} MB."
            ),
        )

    return audio_bytes


async def get_queued_count() -> int:
    value = await redis_get("pipeline:queued")

    if value is None:
        return 0

    return int(value)


async def get_active_count() -> int:
    value = await redis_get("pipeline:active")

    if value is None:
        return 0

    return int(value)


async def ensure_queue_has_capacity(
    requested_files: int,
) -> None:
    queued = await get_queued_count()

    if queued + requested_files > settings.QUEUE_SIZE:
        retry_after = settings.BACKPRESSURE_RETRY_AFTER_SECONDS

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "BACKPRESSURE",
                "message": (
                    f"System is busy. Please retry after {retry_after} seconds."
                ),
                "queue_size": settings.QUEUE_SIZE,
                "queued": queued,
                "requested_files": requested_files,
                "retry_after_seconds": retry_after,
            },
            headers={
                "Retry-After": str(retry_after),
            },
        )


async def get_audio_queue_status() -> QueueStatusResponse:
    queued = await get_queued_count()
    active = await get_active_count()

    return QueueStatusResponse(
        pipeline_limit=settings.PIPELINE_WORKERS,
        active=active,
        queue_limit=settings.QUEUE_SIZE,
        queued=queued,
        available_queue_slots=max(
            0,
            settings.QUEUE_SIZE - queued,
        ),
    )


async def get_audio_batch_status(
    batch_id: str,
) -> AudioBatchStatusResponse:
    batch_key = build_batch_key(batch_id)
    batch_files_key = build_batch_files_key(batch_id)

    batch_data = await redis_hgetall(batch_key)

    if not batch_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch not found: {batch_id}",
        )

    file_ids = await redis_lrange(
        batch_files_key,
        0,
        -1,
    )

    decoded_batch_data = decode_redis_hash(batch_data)

    files: list[AudioFileResponse] = []

    queued_files = 0
    processing_files = 0
    successful_files = 0
    failed_files = 0

    for raw_file_id in file_ids:
        file_id = decode_redis_value(raw_file_id)
        file_key = build_file_key(batch_id, file_id)

        file_data = await redis_hgetall(file_key)

        if not file_data:
            continue

        decoded_data = decode_redis_hash(file_data)

        file_status = normalize_audio_status(decoded_data.get("status"))

        if file_status == "queued":
            queued_files += 1
        elif file_status == "processing":
            processing_files += 1
        elif file_status == "success":
            successful_files += 1
        elif file_status == "failed":
            failed_files += 1

        files.append(
            AudioFileResponse(
                file_id=decoded_data.get("file_id", file_id),
                filename=decoded_data.get("filename", ""),
                status=file_status,
                text=decoded_data.get("text") or None,
                error=decoded_data.get("error") or None,
            )
        )

    if decoded_batch_data.get("status") == "cancelled":
        batch_status = "cancelled"
    else:
        batch_status = calculate_batch_status(
            total_files=len(files),
            processing_files=processing_files,
            successful_files=successful_files,
            failed_files=failed_files,
        )

    return AudioBatchStatusResponse(
        batch_id=batch_id,
        status=batch_status,
        total_files=len(files),
        queued_files=queued_files,
        processing_files=processing_files,
        successful_files=successful_files,
        failed_files=failed_files,
        files=files,
    )


async def wait_for_audio_batch_result(
    batch_id: str,
) -> AudioBatchStatusResponse:
    deadline = time.monotonic() + settings.UPLOAD_WAIT_TIMEOUT_SECONDS

    while True:
        batch_status = await get_audio_batch_status(batch_id)

        if batch_status.status in TERMINAL_BATCH_STATUSES:
            failed_files = [
                file for file in batch_status.files if file.status == "failed"
            ]

            if failed_files:
                logger.error(
                    "Audio batch finished with failed files. batch_id=%s errors=%s",
                    batch_id,
                    [file.error for file in failed_files],
                )

            return batch_status

        if time.monotonic() >= deadline:
            logger.error(
                "Timed out waiting for audio batch result. batch_id=%s status=%s queued=%s processing=%s",
                batch_id,
                batch_status.status,
                batch_status.queued_files,
                batch_status.processing_files,
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={
                    "code": "TRANSCRIPTION_TIMEOUT",
                    "message": "Timed out waiting for transcription result.",
                    "batch_id": batch_id,
                    "current_status": batch_status.model_dump(),
                },
            )

        await asyncio.sleep(settings.UPLOAD_WAIT_POLL_SECONDS)


def calculate_batch_status(
    *,
    total_files: int,
    processing_files: int,
    successful_files: int,
    failed_files: int,
) -> BatchStatus:
    if total_files == 0:
        return "failed"

    if successful_files == total_files:
        return "completed"

    if failed_files == total_files:
        return "failed"

    if failed_files > 0:
        return "partial_failed"

    if processing_files > 0:
        return "processing"

    return "queued"


async def cancel_audio_batch(
    batch_id: str,
) -> dict[str, str]:
    batch_key = build_batch_key(batch_id)
    batch_files_key = build_batch_files_key(batch_id)

    exists = await redis_exists(batch_key)

    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch not found: {batch_id}",
        )

    await hset_mapping(
        batch_key,
        {
            "status": "cancelled",
        },
    )

    file_ids = await redis_lrange(
        batch_files_key,
        0,
        -1,
    )

    for raw_file_id in file_ids:
        file_id = decode_redis_value(raw_file_id)
        file_key = build_file_key(batch_id, file_id)

        await hset_mapping(
            file_key,
            {
                "status": "cancelled",
            },
        )

    return {
        "batch_id": batch_id,
        "status": "cancelled",
    }
