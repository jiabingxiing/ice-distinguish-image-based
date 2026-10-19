"""服务错误码（§16.3）。

400 参数/解码错误 · 422 契约校验失败 · 503 模型未就绪 · 500 内部错误（不泄露堆栈）
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    BAD_REQUEST = "bad_request"
    INVALID_CONTRACT = "invalid_contract"
    MODEL_NOT_READY = "model_not_ready"
    IMAGE_DECODE_FAILED = "image_decode_failed"
    INTERNAL = "internal_error"


class ServiceError(Exception):
    def __init__(self, code: ErrorCode, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class ImageDecodeError(ServiceError):
    def __init__(self, message: str = "图像解码失败") -> None:
        super().__init__(ErrorCode.IMAGE_DECODE_FAILED, message, 400)


class ModelNotReadyError(ServiceError):
    def __init__(self, message: str = "模型未就绪") -> None:
        super().__init__(ErrorCode.MODEL_NOT_READY, message, 503)
