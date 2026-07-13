from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-transcribe"

    # Workers / Queue
    UVICORN_WORKERS: int
    PIPELINE_WORKERS: int
    CPU_WORKERS: int
    OPENAI_CONCURRENCY: int
    QUEUE_SIZE: int

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # TTL
    AUDIO_TTL_SECONDS: int
    RESULT_TTL_SECONDS: int

    # Upload limit
    MAX_AUDIO_FILES: int
    MAX_AUDIO_SIZE_MB: int

    # WebRTC VAD
    WEBRTC_VAD_MODE: int
    VAD_FRAME_MS: int
    VAD_END_SILENCE_MS: int
    VAD_PADDING_MS: int

    # Silero VAD
    SILERO_THRESHOLD: float
    SILERO_FRAME_SIZE: int


settings = Settings()  # type: ignore
