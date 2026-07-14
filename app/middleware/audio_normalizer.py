import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.core.errors import AudioNormalizationError


@dataclass(slots=True)
class NormalizedAudio:
    """
    Kết quả sau khi chuẩn hóa audio.

    pcm_bytes:
        Raw PCM16 bytes.

    sample_rate:
        Sample rate sau chuẩn hóa. Mặc định là 16000 Hz.

    channels:
        Số channel sau chuẩn hóa. Mặc định là mono = 1.

    sample_width:
        Số byte cho mỗi sample. PCM16 = 2 bytes.

    duration_seconds:
        Thời lượng audio tính theo giây.
    """

    pcm_bytes: bytes
    sample_rate: int
    channels: int
    sample_width: int
    duration_seconds: float


def normalize_audio(
    audio_bytes: bytes,
    filename: str | None = None,
) -> NormalizedAudio:
    """
    Decode audio gốc và chuẩn hóa thành PCM16 mono 16 kHz.

    Input có thể là mp3, m4a, wav, flac...
    Output là raw PCM16 bytes.

    Ưu tiên xử lý qua stdin/stdout. Với một số container như m4a/mp4,
    ffmpeg cần input seekable nên sẽ fallback qua temp file trong /tmp.
    """

    if not audio_bytes:
        raise AudioNormalizationError("Audio bytes is empty.")

    suffix = Path(filename or "audio").suffix.lower()

    try:
        completed_process = run_ffmpeg_normalize(
            input_ref="pipe:0",
            input_bytes=audio_bytes,
        )

    except subprocess.CalledProcessError as pipe_exc:
        if suffix not in {".m4a", ".mp4", ".mov"}:
            raise build_ffmpeg_error(pipe_exc) from pipe_exc

        completed_process = run_ffmpeg_normalize_with_temp_file(
            audio_bytes=audio_bytes,
            suffix=suffix,
        )

    pcm_bytes = completed_process.stdout

    if not pcm_bytes:
        if suffix in {".m4a", ".mp4", ".mov"}:
            completed_process = run_ffmpeg_normalize_with_temp_file(
                audio_bytes=audio_bytes,
                suffix=suffix,
            )
            pcm_bytes = completed_process.stdout

        if not pcm_bytes:
            stderr = completed_process.stderr.decode(
                "utf-8",
                errors="ignore",
            ).strip()
            raise AudioNormalizationError(
                "FFmpeg returned empty PCM audio."
                + (f" stderr={stderr}" if stderr else "")
            )

    duration_seconds = calculate_pcm_duration_seconds(
        pcm_bytes=pcm_bytes,
        sample_rate=settings.SAMPLE_RATE,
        channels=settings.CHANNELS,
        sample_width=settings.SAMPLE_WIDTH,
    )

    return NormalizedAudio(
        pcm_bytes=pcm_bytes,
        sample_rate=settings.SAMPLE_RATE,
        channels=settings.CHANNELS,
        sample_width=settings.SAMPLE_WIDTH,
        duration_seconds=duration_seconds,
    )


def run_ffmpeg_normalize(
    *,
    input_ref: str,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_ref,
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        str(settings.CHANNELS),
        "-ar",
        str(settings.SAMPLE_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ]

    return subprocess.run(
        command,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def run_ffmpeg_normalize_with_temp_file(
    *,
    audio_bytes: bytes,
    suffix: str,
) -> subprocess.CompletedProcess[bytes]:
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix or ".audio") as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()

            return run_ffmpeg_normalize(
                input_ref=temp_file.name,
            )

    except subprocess.CalledProcessError as exc:
        raise build_ffmpeg_error(exc) from exc


def build_ffmpeg_error(
    exc: subprocess.CalledProcessError,
) -> AudioNormalizationError:
    error_message = exc.stderr.decode(
        "utf-8",
        errors="ignore",
    ).strip()

    return AudioNormalizationError(f"FFmpeg failed to normalize audio: {error_message}")


def calculate_pcm_duration_seconds(
    *,
    pcm_bytes: bytes,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> float:
    """
    Tính thời lượng audio từ raw PCM bytes.
    """

    bytes_per_second = sample_rate * channels * sample_width

    return len(pcm_bytes) / bytes_per_second
