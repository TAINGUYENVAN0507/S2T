class AppError(Exception):
    """
    Lỗi gốc của toàn bộ project.

    Các lỗi custom khác nên kế thừa từ class này
    để dễ catch chung nếu cần.
    """

    def __init__(
        self,
        message: str,
        code: str = "APP_ERROR",
    ) -> None:
        self.message = message
        self.code = code
        super().__init__(message)


class QueueFullError(AppError):
    """
    Lỗi khi queue đã đầy.
    """

    def __init__(
        self,
        message: str = "Audio processing queue is full.",
    ) -> None:
        super().__init__(message=message, code="QUEUE_FULL")


class AudioNotFoundError(AppError):
    """
    Lỗi khi không tìm thấy audio trong Redis.
    """

    def __init__(
        self,
        message: str = "Audio data was not found.",
    ) -> None:
        super().__init__(
            message=message,
            code="AUDIO_NOT_FOUND",
        )


class UnsupportedAudioFormatError(AppError):
    """
    Lỗi khi file audio không đúng định dạng hỗ trợ.
    """

    def __init__(
        self,
        message: str = "Unsupported audio format.",
    ) -> None:
        super().__init__(
            message=message,
            code="UNSUPPORTED_AUDIO_FORMAT",
        )


class AudioTooLargeError(AppError):
    """
    Lỗi khi file audio vượt quá kích thước cho phép.
    """

    def __init__(
        self,
        message: str = "Audio file is too large.",
    ) -> None:
        super().__init__(
            message=message,
            code="AUDIO_TOO_LARGE",
        )


class AudioNormalizationError(AppError):
    """
    Lỗi khi decode/resample audio thất bại.
    """

    def __init__(
        self,
        message: str = "Failed to normalize audio.",
    ) -> None:
        super().__init__(
            message=message,
            code="AUDIO_NORMALIZATION_FAILED",
        )


class VADProcessingError(AppError):
    """
    Lỗi khi chạy WebRTC VAD hoặc Silero VAD.
    """

    def __init__(
        self,
        message: str = "Failed to process VAD.",
    ) -> None:
        super().__init__(
            message=message,
            code="VAD_PROCESSING_FAILED",
        )


class TranscriptionError(AppError):
    """
    Lỗi khi gọi OpenAI speech-to-text thất bại.
    """

    def __init__(
        self,
        message: str = "Failed to transcribe audio.",
    ) -> None:
        super().__init__(
            message=message,
            code="TRANSCRIPTION_FAILED",
        )
