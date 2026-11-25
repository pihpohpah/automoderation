#!/usr/bin/env python3
"""Паритет: ONNX-артефакты (SmolVLM2) против эталонного PyTorch/HF (models/trained_model).

    uv run --group tools python -m tools.verify_smolvlm --artifacts artifacts/smolvlm_fp32 --tol 0.1

Использует реальный HF-процессор (apply_chat_template,
do_image_splitting=False) для референса и напрямую кормит те же pixel_values/input_ids в
ONNX-графы — так изолируется корректность экспорта (wrappers_smolvlm.py) от numpy-препроцессинга
сервиса, который проверяется отдельно.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CATEGORY_KEYS = ["dangerous", "spam", "junk", "mismatch"]


def make_cases(rng: np.random.Generator):
    def rnd(h, w):
        return Image.fromarray((rng.random((h, w, 3)) * 255).astype("uint8"))

    def grad(h, w):
        g = np.tile(np.linspace(0, 255, w).astype("uint8"), (h, 1))
        return Image.fromarray(np.stack([g, g[::-1], g], -1))

    return [
        ("text_only", "Классный концерт в субботу, приходите в парк!", []),
        ("empty_text_1img", "", [grad(300, 500)]),
        ("1img", "Кафе на углу, уютно и вкусно", [rnd(480, 640)]),
        ("4img", "Гулял по городу, нашёл классное место", [rnd(300, 400), grad(256, 256), rnd(700, 300), grad(500, 500)]),
    ]


def compute_cv_features(images: list[Image.Image]) -> np.ndarray:
    from scipy.ndimage import laplace

    if not images:
        return np.zeros(9, dtype=np.float32)
    stats = []
    for img in images:
        gray = np.asarray(img.convert("L"), dtype=np.float64)
        blur = float(laplace(gray).var())
        log_blur = float(np.log1p(blur))
        brightness = float(gray.mean() / 255.0)
        contrast = float(gray.std() / 255.0)
        hist, _ = np.histogram(gray, bins=64, range=(0, 255), density=True)
        hist = hist[hist > 0]
        entropy = float(-(hist * np.log2(hist)).sum()) if len(hist) else 0.0
        stats.append([log_blur, brightness, contrast, entropy])
    stats = np.array(stats, dtype=np.float32)
    return np.concatenate([stats.mean(0), stats.min(0), [float(len(images))]]).astype(np.float32)


@torch.no_grad()
def reference(model, processor, text, images, head_w, head_b):
    content = [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": text}]
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        [messages], add_generation_prompt=False, tokenize=True, return_dict=True,
        return_tensors="pt", padding=True, do_image_splitting=False,
    )
    inputs.pop("pixel_attention_mask", None)
    model_kwargs = dict(inputs)
    if "pixel_values" in model_kwargs and model_kwargs["pixel_values"].numel() == 0:
        model_kwargs.pop("pixel_values")
    out = model(**model_kwargs, output_hidden_states=True, use_cache=False)
    hid = out.hidden_states[-1][0, -1].numpy()
    cv = compute_cv_features(images)
    logits = head_w @ np.concatenate([hid, cv]) + head_b
    return logits, inputs


def run_onnx(vis_sess, lm_sess, emb, inputs, head_w, head_b, cv):
    input_ids = inputs["input_ids"][0].numpy()
    pixel_values = inputs.get("pixel_values")

    x = emb[input_ids].astype(np.float32)
    if pixel_values is not None and pixel_values.numel() > 0:
        pv = pixel_values[0].numpy().astype(np.float32)  # [n_images,3,512,512]
        img_feats = vis_sess.run(None, {"pixel_values": pv})[0]  # [n_images*64, hidden]
        mask = input_ids == 49190
        assert mask.sum() == img_feats.shape[0], (mask.sum(), img_feats.shape)
        x[mask] = img_feats

    T = x.shape[0]
    head_dim = 64
    rope_theta = 100000.0
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    pos = np.arange(T, dtype=np.float32)
    freqs = pos[:, None] * inv_freq[None, :]
    emb_rope = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = np.cos(emb_rope).astype(np.float32), np.sin(emb_rope).astype(np.float32)

    hidden = lm_sess.run(None, {"inputs_embeds": x[None], "cos": cos, "sin": sin})[0]  # [1,D]
    feats = np.concatenate([hidden[0], cv])
    return head_w @ feats + head_b


def run_engine_logits(eng, text, imgs):
    res = eng.predict(text, imgs)
    sig_inv = lambda p: np.log(p / (1 - p))  # noqa: E731
    return np.array([sig_inv(res.probs[c]) for c in CATEGORY_KEYS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--model-dir", default=str(ROOT / "models/trained_model"))
    ap.add_argument("--tokenizer-dir", default=str(ROOT / "models/SmolVLM2-256M"))
    ap.add_argument("--mode", choices=["onnx", "engine"], default="engine",
                    help="onnx: сравнить ONNX-графы напрямую на тензорах HF-процессора; "
                         "engine: сквозная проверка — весь service/app (numpy-препроцессинг + Engine)")
    ap.add_argument("--tol", type=float, default=None)
    a = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    art = Path(a.artifacts)
    model = AutoModelForImageTextToText.from_pretrained(a.model_dir, dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(a.tokenizer_dir)
    head = np.load(art / "head.npz")
    head_w, head_b = head["W"].astype(np.float32), head["b"].astype(np.float32)

    if a.mode == "onnx":
        emb = np.load(art / "embed_tokens.npy").astype(np.float32)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        vis_sess = ort.InferenceSession(str(art / "vision.onnx"), so, providers=["CPUExecutionProvider"])
        lm_sess = ort.InferenceSession(str(art / "lm.onnx"), so, providers=["CPUExecutionProvider"])
    else:
        sys.path.insert(0, str(ROOT / "service"))
        from app.engine import Engine, EngineConfig  # noqa: E402

        eng = Engine(EngineConfig(artifacts_dir=art))

    worst = 0.0
    for name, text, imgs in make_cases(np.random.default_rng(0)):
        ref_logits, inputs = reference(model, processor, text, imgs, head_w, head_b)
        if a.mode == "onnx":
            cv = compute_cv_features(imgs)
            got_logits = run_onnx(vis_sess, lm_sess, emb, inputs, head_w, head_b, cv)
        else:
            got_logits = run_engine_logits(eng, text, imgs)
        d = float(np.abs(ref_logits - got_logits).max())
        worst = max(worst, d)
        sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
        dec_ok = bool(((sig(ref_logits) > 0.5) == (sig(got_logits) > 0.5)).all())
        print(f"{name:16s} T={inputs['input_ids'].shape[1]:4d} max|dlogit|={d:.5f} decisions_equal={dec_ok}")
    print(f"WORST max|dlogit| = {worst:.5f}")
    if a.tol is not None and worst > a.tol:
        print("FAIL: превышен допуск")
        sys.exit(1)


if __name__ == "__main__":
    main()
