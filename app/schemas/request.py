from pydantic import BaseModel, Field


class AudioUploadOptions(BaseModel):
    """
    Các tùy chọn xử lý audio.

    Class này không bắt buộc phải dùng ngay trong POST /audios/.
    Nó hữu ích nếu sau này muốn nhận JSON config riêng.
    """

    language: str = Field(
        default="vi",
        description="Ngôn ngữ chính của audio. Ví dụ: vi, en.",
    )

    use_vad: bool = Field(
        default=True,
        description="Có sử dụng WebRTC + Silero VAD hay không.",
    )


class CancelBatchRequest(BaseModel):
    """
    Request body dùng khi muốn hủy một batch.

    Nếu endpoint DELETE /audios/{batch_id} không cần body
    thì class này có thể chưa dùng đến.
    """

    reason: str | None = Field(
        default=None,
        description="Lý do hủy batch, nếu có.",
    )
