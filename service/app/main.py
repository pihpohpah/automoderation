"""FastAPI-приложение автомодерации.

POST /moderate  (multipart/form-data): text (str), images (0..N файлов) -> JSON с вероятностями и вердиктом.
GET  /health    — готовность (модель загружена и прогрета).

Ограничения под 1 CPU / ~1-2 ГБ RAM / ~10 с: инференс строго последовательный (один за раз),
очередь ограничена, общий дедлайн на запрос, лимиты на размер/число картинок.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

from .engine import DeadlineExceeded, Engine, EngineConfig

log = logging.getLogger("moderation")

Image.MAX_IMAGE_PIXELS = 60_000_000  # защита от decompression bomb (PIL бросит DecompressionBombError выше x2)


def _env(name: str, default, cast=str):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


class Settings:
    artifacts_dir = Path(_env("ARTIFACTS_DIR", "artifacts/smolvlm_int8"))
    max_images = _env("MAX_IMAGES", 4, int)
    max_image_bytes = _env("MAX_IMAGE_BYTES", 8 * 1024 * 1024, int)
    max_text_chars = _env("MAX_TEXT_CHARS", 4000, int)
    timeout_s = _env("TIMEOUT_S", 10.0, float)            # общий бюджет на запрос (включая ожидание в очереди)
    safety_margin_s = _env("SAFETY_MARGIN_S", 0.4, float)  # запас на сериализацию ответа/сеть
    max_queue = _env("MAX_QUEUE", 2, int)                  # сколько запросов могут ждать, пока идёт расчёт
    max_text_tokens = _env("MAX_TEXT_TOKENS", 512, int)
    threads = _env("ORT_THREADS", 1, int)
    warmup = _env("WARMUP", "1") == "1"
    thresholds = {
        # 0.75 вместо 0.5 — дефолтный порог давал много ложных срабатываний
        "dangerous": _env("THR_DANGEROUS", 0.75, float),
        "spam": _env("THR_SPAM", 0.5, float),
        "junk": _env("THR_JUNK", 0.5, float),
        "mismatch": _env("THR_MISMATCH", 0.5, float),
    }


S = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.perf_counter()
    app.state.engine = Engine(EngineConfig(artifacts_dir=S.artifacts_dir, max_images=S.max_images,
                                           max_text_tokens=S.max_text_tokens, intra_threads=S.threads,
                                           thresholds=S.thresholds))
    if S.warmup:  # прогреваем ORT (аллокации, JIT-ядра), чтобы первый реальный запрос не съел бюджет
        rng = np.random.default_rng(0)
        img = Image.fromarray((rng.random((448, 448, 3)) * 255).astype("uint8"))
        app.state.engine.predict("прогрев", [img])
    app.state.sem = asyncio.Semaphore(1)
    app.state.waiting = 0
    app.state.ready = True
    log.info("engine ready in %.1fs (%s)", time.perf_counter() - t, S.artifacts_dir)
    yield


app = FastAPI(title="Content moderation service", version="0.1.0", lifespan=lifespan)


def _load_image(data: bytes, name: str) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.draft("RGB", (1024, 1024))               # JPEG: декодируем сразу в уменьшенном виде (быстро и дёшево по памяти)
        img = ImageOps.exif_transpose(img)           # (EXIF у датасета удалён -> ориентацию применяем и забываем)
        img = img.convert("RGB")
        img.thumbnail((1024, 1024), Image.BICUBIC)   # как в препроцессинге датасета: большая сторона <= 1024
        return img
    except Image.DecompressionBombError:
        raise HTTPException(413, f"image '{name}' is too large (pixels)")
    except Exception:
        raise HTTPException(422, f"image '{name}' is not a valid image")


def _verdict(probs: dict[str, float]) -> dict:
    cats = {k: {"prob": round(p, 5), "flag": p >= S.thresholds[k]} for k, p in probs.items()}
    flagged = [k for k, v in cats.items() if v["flag"]]
    return {"label": 1 if flagged else 0, "flagged": flagged, "categories": cats}


def _rss_mb() -> tuple[float, float]:
    """(текущий RSS, пиковый RSS) в МБ — без psutil."""
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = peak / (1024 * 1024) if os.uname().sysname == "Darwin" else peak / 1024
    try:
        with open("/proc/self/statm") as f:
            cur = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        cur = None
    return (round(cur, 1) if cur is not None else None), round(peak, 1)


@app.get("/health")
async def health():
    cur, peak = _rss_mb()
    return {"status": "ok" if getattr(app.state, "ready", False) else "loading", "rss_mb": cur, "peak_rss_mb": peak}


@app.post("/moderate")
async def moderate(text: str = Form(""), images: list[UploadFile] = File(default_factory=list)):
    t_start = time.monotonic()
    deadline = t_start + S.timeout_s - S.safety_margin_s
    if len(text) > S.max_text_chars:
        raise HTTPException(413, f"text is longer than {S.max_text_chars} chars")
    # Пустые файловые части (клиент прислал пустое поле) игнорируем
    images = [f for f in images if f.filename or f.size]
    if len(images) > S.max_images:
        raise HTTPException(422, f"at most {S.max_images} images allowed")
    raw: list[tuple[str, bytes]] = []
    for f in images:
        data = await f.read(S.max_image_bytes + 1)
        if len(data) > S.max_image_bytes:
            raise HTTPException(413, f"image '{f.filename}' exceeds {S.max_image_bytes} bytes")
        raw.append((f.filename or "image", data))

    engine: Engine = app.state.engine
    if app.state.waiting >= S.max_queue:
        raise HTTPException(503, "server is busy, retry later", headers={"Retry-After": "2"})
    app.state.waiting += 1
    try:
        try:
            await asyncio.wait_for(app.state.sem.acquire(), timeout=max(0.0, deadline - time.monotonic()))
        except asyncio.TimeoutError:
            raise HTTPException(503, "server is busy, retry later", headers={"Retry-After": "2"})
    finally:
        app.state.waiting -= 1
    try:
        def work():
            pil = [_load_image(d, n) for n, d in raw]
            return engine.predict(text, pil, deadline=deadline)

        try:
            res = await asyncio.get_running_loop().run_in_executor(None, work)
        except DeadlineExceeded:
            raise HTTPException(504, "inference deadline exceeded")
    finally:
        app.state.sem.release()

    body = _verdict(res.probs)
    body["timings_ms"] = {k: round(v, 1) for k, v in res.timings_ms.items()}
    body["timings_ms"]["queue_and_io"] = round((time.monotonic() - t_start) * 1000 - res.timings_ms["total"], 1)
    body["n_images"] = res.n_images
    body["n_tokens"] = res.n_tokens
    return JSONResponse(body)
