from fastapi import APIRouter, File, Query, UploadFile, status

from app.middleware.audio_job import (
    create_audio_batch,
    get_audio_queue_status,
    wait_for_audio_batch_result,
)
from app.schemas.response import (
    AudioBatchStatusResponse,
    QueueStatusResponse,
)

router = APIRouter(
    prefix="/audios",
    tags=["Audios"],
)


@router.post(
    "/",
    response_model=AudioBatchStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_audios(
    files: list[UploadFile] = File(...),
    language: str = Query(default="vi"),
    use_vad: bool = Query(default=True),
) -> AudioBatchStatusResponse:
    """
    Upload một hoặc nhiều file audio.

    Upload 1 file chỉ là trường hợp đặc biệt của upload nhiều file.
    """
    created_batch = await create_audio_batch(
        files=files,
        language=language,
        use_vad=use_vad,
    )

    return await wait_for_audio_batch_result(created_batch.batch_id)


@router.get(
    "/queue/status",
    response_model=QueueStatusResponse,
)
async def get_queue_status() -> QueueStatusResponse:
    """
    Lấy trạng thái queue hiện tại.
    """
    return await get_audio_queue_status()
