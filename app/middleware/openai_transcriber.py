import io
import wave

from openai import OpenAI

from app.config import settings
from app.core.errors import TranscriptionError


client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
)


def pcm_to_wav_bytes(
    *,
    pcm_bytes: bytes,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> bytes:
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)

    return wav_buffer.getvalue()


def transcribe_pcm(
    *,
    pcm_bytes: bytes,
    sample_rate: int,
    channels: int,
    sample_width: int,
    language: str,
) -> str:
    if not pcm_bytes:
        return ""

    wav_bytes = pcm_to_wav_bytes(
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )

    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = "audio.wav"

    try:
        transcription = client.audio.transcriptions.create(
            model=settings.OPENAI_MODEL,
            file=audio_file,
            language=language,
        )

    except Exception as exc:
        raise TranscriptionError(f"OpenAI transcription failed: {exc}") from exc

    text = getattr(transcription, "text", transcription)

    return str(text).strip()
