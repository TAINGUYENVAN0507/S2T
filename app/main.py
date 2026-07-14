from fastapi import FastAPI

from app.api.audio_router import router as audio_router
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(
    title="S2T Realtime Audio API",
    version="0.1.0",
)
app.include_router(audio_router)
