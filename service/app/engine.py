"""ONNX-движок модерации (SmolVLM2/Idefics3): vision.onnx + lm.onnx + embed_tokens.npy (mmap) + линейная голова."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

from . import preprocess as P


@dataclass
class EngineConfig:
    artifacts_dir: Path
    max_images: int = 4
    max_text_tokens: int = 512
    intra_threads: int = 1
    thresholds: dict[str, float] = field(default_factory=lambda: {
        "dangerous": 0.5, "spam": 0.5, "junk": 0.5, "mismatch": 0.5})


class DeadlineExceeded(Exception):
    pass


@dataclass
class Result:
    probs: dict[str, float]
    timings_ms: dict[str, float]
    n_tokens: int
    n_images: int
    hidden: np.ndarray | None = None


def _session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False       # меньше пиковой памяти при динамических формах
    so.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class Engine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        d = cfg.artifacts_dir
        self.meta = json.loads((d / "meta.json").read_text())
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.embed = np.load(d / "embed_tokens.npy", mmap_mode="r")
        self.embed_scale = np.load(d / "embed_scale.npy", mmap_mode="r") if self.meta.get("embed_dtype") == "int8" else None
        head = np.load(d / "head.npz")
        self.head_w, self.head_b = head["W"].astype(np.float32), head["b"].astype(np.float32)
        self.vis = _session(d / "vision.onnx", cfg.intra_threads)
        self.lm = _session(d / "lm.onnx", cfg.intra_threads)
        self.categories: list[str] = self.meta["categories"]
        self.text_inv = np.asarray(self.meta["text_inv_freq"], np.float32)
        self.image_size = int(self.meta["image_size"])
        self.tokens_per_image = int(self.meta["tokens_per_image"])
        self.image_token_id = int(self.meta["image_token_id"])

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _run(sess: ort.InferenceSession, feeds: dict, deadline: float | None):
        """sess.run с жёстким дедлайном: по таймеру выставляем RunOptions.terminate -> ORT прерывает граф."""
        if deadline is None:
            return sess.run(None, feeds)
        left = deadline - time.monotonic()
        if left <= 0:
            raise DeadlineExceeded()
        ro = ort.RunOptions()
        timer = threading.Timer(left, lambda: setattr(ro, "terminate", True))
        timer.start()
        try:
            return sess.run(None, feeds, run_options=ro)
        except Exception as e:  # ORT бросает RuntimeException "Exiting due to terminate flag being set to true"
            if ro.terminate or (time.monotonic() >= deadline):
                raise DeadlineExceeded() from e
            raise
        finally:
            timer.cancel()

    def _truncate_text(self, text: str) -> str:
        """Обрезка по токенам (лимит max_text_tokens), а не по символам — но сам текст должен
        остаться строкой: промпт целиком токенизируется одним вызовом encode() (см. preprocess.py)."""
        ids = self.tok.encode(text or "", add_special_tokens=False).ids
        if len(ids) <= self.cfg.max_text_tokens:
            return text or ""
        return self.tok.decode(ids[: self.cfg.max_text_tokens])

    def _encode_prompt(self, text: str, n_images: int) -> list[int]:
        text = self._truncate_text(text)
        s = P.build_prompt_string(text, n_images, self.tokens_per_image)
        return self.tok.encode(s, add_special_tokens=False).ids

    def _embed(self, ids: np.ndarray) -> np.ndarray:
        rows = np.asarray(self.embed[ids], dtype=np.float32)
        if self.embed_scale is not None:
            rows *= np.asarray(self.embed_scale[ids], dtype=np.float32)[:, None]
        return rows

    # ------------------------------------------------------------------ main
    def predict(self, text: str, images: list[Image.Image], deadline: float | None = None) -> Result:
        """deadline — абсолютное время time.monotonic(), после которого расчёт прерывается (DeadlineExceeded)."""
        cfg, meta = self.cfg, self.meta
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        images = images[: cfg.max_images]

        # --- CV-признаки считаем по изображениям, уже приведённым к рабочему размеру (экономит CPU)
        cv = P.compute_cv_features(images)
        timings["cv_features"] = (time.perf_counter() - t0) * 1000

        # --- vision: все картинки одним батчем (одинаковый фиксированный размер image_size x image_size)
        t1 = time.perf_counter()
        img_feats = None
        if images:
            pv = np.stack([P.image_to_pixel_values(img, self.image_size, meta["image_mean"], meta["image_std"])
                          for img in images]).astype(np.float32)
            img_feats = self._run(self.vis, {"pixel_values": pv}, deadline)[0]  # [n_images*tokens_per_image, D]
        timings["vision"] = (time.perf_counter() - t1) * 1000

        # --- prompt
        t2 = time.perf_counter()
        ids = np.asarray(self._encode_prompt(text, len(images)), dtype=np.int64)
        T = len(ids)
        is_img = ids == self.image_token_id
        x = self._embed(np.where(is_img, 0, ids))
        if img_feats is not None:
            assert is_img.sum() == img_feats.shape[0], (is_img.sum(), img_feats.shape)
            x[is_img] = img_feats
        cos, sin = P.text_rope(T, self.text_inv)
        timings["prompt"] = (time.perf_counter() - t2) * 1000

        # --- LM
        t3 = time.perf_counter()
        hidden = self._run(self.lm, {"inputs_embeds": x[None], "cos": cos, "sin": sin}, deadline)[0]  # [1,D]
        timings["lm"] = (time.perf_counter() - t3) * 1000

        feats = np.concatenate([hidden[0].astype(np.float32), cv])
        logits = self.head_w @ feats + self.head_b
        probs = 1.0 / (1.0 + np.exp(-logits))
        timings["total"] = (time.perf_counter() - t0) * 1000
        return Result({c: float(p) for c, p in zip(self.categories, probs)}, timings, T, len(images), hidden[0])
