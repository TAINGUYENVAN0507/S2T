import logging
import sys
from pathlib import Path

from app.config import settings


def setup_logging() -> None:
    """
    Cấu hình logging cho toàn project.

    Log được ghi vào một file cố định:
        logs/app.log

    Mỗi dòng log đều có ngày giờ để dễ tìm kiếm.
    """

    log_dir = Path(settings.LOG_DIR)

    # Tạo thư mục logs nếu chưa có.
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = log_dir / settings.LOG_FILE

    log_level = getattr(
        logging,
        settings.LOG_LEVEL.upper(),
        logging.INFO,
    )

    log_format = (
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(filename)s:%(lineno)d | "
        "%(message)s"
    )

    date_format = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()

    # Tránh bị add handler nhiều lần khi uvicorn reload
    # hoặc khi setup_logging() bị gọi ở nhiều nơi.
    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.setLevel(log_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        logging.Formatter(
            fmt=log_format,
            datefmt=date_format,
        )
    )

    file_handler = logging.FileHandler(
        filename=log_file,
        mode="a",  # append, không ghi đè log cũ
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter(
            fmt=log_format,
            datefmt=date_format,
        )
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.error").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(log_level)
    logging.getLogger("celery").setLevel(log_level)
