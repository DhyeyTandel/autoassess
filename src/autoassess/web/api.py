"""FastAPI app: POST /api/assess, GET /api/health, and the built frontend.

Inference is CPU/GPU-bound and the YOLO models are shared, so one assessment
runs at a time (an asyncio.Lock around a threadpool call).

Env: AUTOASSESS_HOST (127.0.0.1), AUTOASSESS_PORT (7860), AUTOASSESS_DEVICE
(cpu), AUTOASSESS_MAX_UPLOAD_MB (15), AUTOASSESS_STATIC_DIR (frontend/dist).
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image

from autoassess.utils.logging import get_logger
from autoassess.web import service

log = get_logger(__name__)

DEFAULT_MAX_UPLOAD_MB = 15.0
DEFAULT_STATIC_DIR = "frontend/dist"
_CHUNK = 1024 * 1024
_MULTIPART_SLACK = 64 * 1024  # form framing allowed on top of the file limit


class AssessorLike(Protocol):
    def assess(self, image: Image.Image, sensitivity: str) -> service.AssessResult: ...


def max_upload_bytes(env: Mapping[str, str] | None = None) -> int:
    e = os.environ if env is None else env
    return int(float(e.get("AUTOASSESS_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB)) * 1024 * 1024)


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    """Read the upload in chunks, aborting as soon as it exceeds `limit` bytes."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(_CHUNK):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Image too large; the limit is {limit / (1024 * 1024):g} MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(assessor: AssessorLike | None = None) -> FastAPI:
    app = FastAPI(title="AutoAssess", docs_url=None, redoc_url=None)
    limit = max_upload_bytes()
    app.state.lock = asyncio.Lock()
    app.state.assessor = assessor if assessor is not None else service.Assessor()

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "busy": app.state.lock.locked()}

    @app.post("/api/assess")
    async def assess(
        request: Request,
        image: Annotated[UploadFile, File()],
        sensitivity: Annotated[Literal["high_recall", "standard"], Form()] = "high_recall",
    ) -> dict[str, Any]:
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit + _MULTIPART_SLACK:
            raise HTTPException(status_code=413, detail="Image too large.")
        data = await _read_limited(image, limit)
        try:
            pil = await run_in_threadpool(service.load_image, data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async with app.state.lock:
            start = time.perf_counter()
            try:
                out = await run_in_threadpool(app.state.assessor.assess, pil, sensitivity)
            except FileNotFoundError as exc:
                log.error("Model weights missing: %s", exc)
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        log.info("assessed %dx%d (%s) in %.0f ms", pil.width, pil.height, sensitivity, elapsed_ms)
        response: dict[str, Any] = await run_in_threadpool(
            service.build_response,
            out.result, out.overlay, sensitivity, out.secondary_conf, elapsed_ms,
        )
        return response

    static_dir = Path(os.environ.get("AUTOASSESS_STATIC_DIR", DEFAULT_STATIC_DIR))
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    else:

        @app.get("/")
        async def root() -> dict[str, str]:
            return {"note": f"Frontend not built ({static_dir} missing); API only (dev mode)."}

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("AUTOASSESS_HOST", "127.0.0.1")
    port = int(os.environ.get("AUTOASSESS_PORT", "7860"))
    assessor = service.Assessor()
    assessor.load()  # fail fast on missing weights; models stay resident
    uvicorn.run(create_app(assessor), host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
