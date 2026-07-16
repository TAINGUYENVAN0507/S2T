from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-transcribe"

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8004

    PIPELINE_WORKERS: int = 32
    QUEUE_SIZE: int = 512
    CELERY_CONCURRENCY: int = 1
    BACKPRESSURE_RETRY_AFTER_SECONDS: int = 30
    UPLOAD_WAIT_TIMEOUT_SECONDS: int = 120
    UPLOAD_WAIT_POLL_SECONDS: float = 0.5

    # Redis
    REDIS_URL: str = "redis://localhost:6379/2"

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

    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    SAMPLE_WIDTH: int = 2

    LOG_DIR: str = "app/logs"
    LOG_FILE: str = "app.log"
    LOG_LEVEL: str = "INFO"


settings = Settings()  # type: ignore
