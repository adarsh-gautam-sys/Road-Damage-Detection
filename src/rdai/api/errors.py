"""Uniform API error classes and FastAPI exception handlers (`A11`)."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class APIError(Exception):
    """Structured API error returning `{'error': {'code': ..., 'message': ...}}`."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = str(code)
        self.message = str(message)


def register_exception_handlers(app: FastAPI) -> None:
    """Register uniform JSON error handlers across all 4xx/5xx responses (`A11`)."""

    @app.exception_handler(APIError)
    async def _handle_api_error(_request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = "; ".join(
            f"{'.'.join(str(loc) for loc in err.get('loc', []))}: {err.get('msg', 'invalid')}"
            for err in exc.errors()
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": details or "Request validation failed",
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code_map = {
            400: "BAD_REQUEST",
            404: "NOT_FOUND",
            409: "CONFLICT",
            413: "PAYLOAD_TOO_LARGE",
            415: "UNSUPPORTED_MEDIA_TYPE",
            422: "UNPROCESSABLE_ENTITY",
        }
        err_code = code_map.get(exc.status_code, f"HTTP_{exc.status_code}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": err_code,
                    "message": str(exc.detail or "HTTP error"),
                }
            },
        )
