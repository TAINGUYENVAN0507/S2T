from typing import Literal

from pydantic import BaseModel, Field

AudioStatus = Literal[
    "queued",
    "processing",
    "success",
    "failed",
    "cancelled",
]


BatchStatus = Literal[
    "queued",
    "processing",
    "completed",
    "partial_failed",
    "failed",
    "cancelled",
]


class AudioFileResponse(BaseModel):
    """
    Response đại diện cho một file audio trong batch.
    """

    file_id: str = Field(
        description="ID duy nhất của file trong batch.",
    )

    filename: str = Field(
        description="Tên file audio gốc.",
    )

    status: AudioStatus = Field(
        description="Trạng thái xử lý hiện tại của file.",
    )

    text: str | None = Field(
        default=None,
        description="Kết quả speech-to-text nếu xử lý thành công.",
    )

    error: str | None = Field(
        default=None,
        description="Thông báo lỗi nếu file xử lý thất bại.",
    )


class AudioBatchCreatedResponse(BaseModel):
    """
    Response trả về sau khi upload audio thành công.

    Lúc này audio mới được lưu vào Redis và đưa vào Celery queue,
    chưa speech-to-text xong.
    """

    batch_id: str = Field(
        description="ID của batch upload.",
    )

    status: Literal["queued"] = Field(
        description="Trạng thái ban đầu của batch.",
    )

    total_files: int = Field(
        description="Tổng số file trong batch.",
    )

    files: list[AudioFileResponse] = Field(
        default_factory=list,
        description="Danh sách file trong batch.",
    )


class AudioBatchStatusResponse(BaseModel):
    """
    Response cho endpoint GET /audios/{batch_id}.
    """

    batch_id: str = Field(
        description="ID của batch.",
    )

    status: BatchStatus = Field(
        description="Trạng thái tổng thể của batch.",
    )

    total_files: int = Field(
        description="Tổng số file trong batch.",
    )

    queued_files: int = Field(
        description="Số file đang chờ xử lý.",
    )

    processing_files: int = Field(
        description="Số file đang được xử lý.",
    )

    successful_files: int = Field(
        description="Số file đã xử lý thành công.",
    )

    failed_files: int = Field(
        description="Số file xử lý thất bại.",
    )

    files: list[AudioFileResponse] = Field(
        default_factory=list,
        description="Danh sách chi tiết từng file.",
    )


class QueueStatusResponse(BaseModel):
    """
    Response cho endpoint GET /audios/queue/status.
    """

    pipeline_limit: int = Field(
        description="Giới hạn số file active trong pipeline.",
    )

    active: int = Field(
        description="Số file đang được xử lý.",
    )

    queue_limit: int = Field(
        description="Giới hạn số file đang chờ trong queue.",
    )

    queued: int = Field(
        description="Số file đang chờ xử lý.",
    )

    available_queue_slots: int = Field(
        description="Số slot queue còn trống.",
    )


class ErrorResponse(BaseModel):
    """
    Response lỗi chuẩn của API.
    """

    code: str = Field(
        description="Mã lỗi dạng string.",
    )

    message: str = Field(
        description="Thông báo lỗi ngắn gọn.",
    )

    detail: dict | None = Field(
        default=None,
        description="Thông tin chi tiết bổ sung nếu có.",
    )
