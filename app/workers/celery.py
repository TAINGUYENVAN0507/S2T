import logging
from typing import Any, cast

import redis
from celery import Celery

from app.config import settings
from app.core.logging import setup_logging
from app.middleware.audio_normalizer import normalize_audio
from app.middleware.openai_transcriber import transcribe_pcm

setup_logging()
logger = logging.getLogger(__name__)

celery_app = Celery(
    "audio_pipeline",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

redis_client = redis.Redis.from_url(
    settings.REDIS_URL,
)

vad_detector: Any | None = None
active_started = False

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_expires=settings.RESULT_TTL_SECONDS,
)


def build_file_key(
    batch_id: str,
    file_id: str,
) -> str:
    """
    Key lưu trạng thái của từng file trong Redis.
    """
    return f"file:{batch_id}:{file_id}"


def get_vad_detector():
    global vad_detector

    if vad_detector is None:
        from app.middleware.vad import TwoStageVAD

        vad_detector = TwoStageVAD()

    return vad_detector


@celery_app.task(
    bind=True,
    name="process_audio_task",
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
)
def process_audio_task(
    self,
    batch_id: str,
    file_id: str,
    audio_key: str,
    filename: str,
    language: str = "vi",
    use_vad: bool = True,
) -> dict:
    """
    Task xử lý một file audio.

    Pipeline sau này:
    1. Đọc audio bytes từ Redis.
    2. Chuẩn hóa audio thành PCM16 mono 16 kHz.
    3. Nếu bật VAD, lọc đoạn có tiếng nói.
    4. Gọi OpenAI gpt-4o-transcribe.
    5. Lưu transcript vào Redis.
    6. Xóa audio bytes tạm.
    """

    file_key = build_file_key(
        batch_id=batch_id,
        file_id=file_id,
    )
    active_started = False

    try:
        logger.info(
            "Start processing audio. batch_id=%s file_id=%s filename=%s",
            batch_id,
            file_id,
            filename,
        )

        # Chuyển trạng thái file sang processing.
        redis_client.hset(
            file_key,
            mapping={
                "status": "processing",
                "error": "",
            },
        )

        # Cập nhật counter queue.
        redis_client.decr("pipeline:queued")
        redis_client.incr("pipeline:active")
        active_started = True

        raw_audio_bytes = redis_client.get(audio_key)

        if raw_audio_bytes is None:
            raise FileNotFoundError(f"Audio bytes not found in Redis: {audio_key}")

        audio_bytes = cast(bytes, raw_audio_bytes)

        logger.info(
            "Loaded audio bytes. batch_id=%s file_id=%s bytes=%s use_vad=%s",
            batch_id,
            file_id,
            len(audio_bytes),
            use_vad,
        )

        normalized_audio = normalize_audio(
            audio_bytes,
            filename=filename,
        )

        if use_vad:
            from app.middleware.vad import join_speech_segments

            segments = get_vad_detector().detect(
                pcm_bytes=normalized_audio.pcm_bytes,
                sample_rate=normalized_audio.sample_rate,
            )
            speech_pcm = join_speech_segments(
                segments=segments,
                sample_rate=normalized_audio.sample_rate,
            )
        else:
            speech_pcm = normalized_audio.pcm_bytes

        logger.info(
            "Prepared speech pcm. batch_id=%s file_id=%s bytes=%s duration=%s",
            batch_id,
            file_id,
            len(speech_pcm),
            normalized_audio.duration_seconds,
        )

        text = transcribe_pcm(
            pcm_bytes=speech_pcm,
            sample_rate=normalized_audio.sample_rate,
            channels=normalized_audio.channels,
            sample_width=normalized_audio.sample_width,
            language=language,
        )

        if not text:
            raise ValueError("OpenAI transcription returned empty text.")

        redis_client.hset(
            file_key,
            mapping={
                "status": "success",
                "text": text,
                "error": "",
            },
        )

        logger.info(
            "Finished processing audio. batch_id=%s file_id=%s filename=%s",
            batch_id,
            file_id,
            filename,
        )

        return {
            "batch_id": batch_id,
            "file_id": file_id,
            "filename": filename,
            "status": "success",
            "text": text,
        }

    except Exception as exc:
        logger.exception(
            "Failed to process audio. batch_id=%s file_id=%s filename=%s",
            batch_id,
            file_id,
            filename,
        )

        redis_client.hset(
            file_key,
            mapping={
                "status": "failed",
                "text": "",
                "error": str(exc),
            },
        )

        raise

    finally:
        # Giảm active nếu task đã tăng active trước đó.
        try:
            if active_started:
                redis_client.decr("pipeline:active")
        except Exception:
            logger.exception("Failed to decrease active counter.")

        # Xóa audio gốc sau khi xử lý xong hoặc lỗi.
        try:
            redis_client.delete(audio_key)
        except Exception:
            logger.exception(
                "Failed to delete temporary audio bytes. audio_key=%s",
                audio_key,
            )
