#!/usr/bin/env python3
"""Паритет: ONNX-артефакты (fp32 или квантованные) против эталонного PyTorch/HF (база + merge LoRA + голова).

    python -m tools.verify --artifacts artifacts/int8 --checkpoint runs/lora_v1/checkpoint.pt

Печатает по каждому тест-кейсу: max|Δlogit|, расхождение вероятностей и совпадение решений.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "service"))
from app import preprocess as P  # noqa: E402
from app.engine import Engine, EngineConfig  # noqa: E402
from tools.common import CATEGORY_KEYS, load_base, merge_lora_and_head  # noqa: E402


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


@torch.no_grad()
def reference(model, tok_proc, text, images, head_w, head_b, max_pixels, min_pixels):
    cfg = model.config
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
    t = tok_proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                     add_generation_prompt=False)
    kw = dict(text=[t], return_tensors="pt", images_kwargs={"max_pixels": max_pixels, "min_pixels": min_pixels})
    if images:
        kw["images"] = images
    inp = tok_proc(**kw)
    hid = model(**inp, output_hidden_states=True).hidden_states[-1][0, -1].numpy()
    cv = P.compute_cv_features(images)
    return head_w @ np.concatenate([hid, cv]) + head_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--model-dir", default=str(ROOT / "models/Qwen3.5-0.8B"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tol", type=float, default=None, help="допуск на max|Δlogit| (fail, если превышен)")
    a = ap.parse_args()

    from transformers import AutoProcessor

    model = load_base(a.model_dir)
    head_w, head_b, _ = merge_lora_and_head(model, a.checkpoint, model.config.text_config.hidden_size)
    proc = AutoProcessor.from_pretrained(a.model_dir)
    eng = Engine(EngineConfig(artifacts_dir=Path(a.artifacts)))
    worst = 0.0
    for name, text, imgs in make_cases(np.random.default_rng(0)):
        ref = reference(model, proc, text, imgs, head_w, head_b, eng.cfg.max_pixels, eng.cfg.min_pixels)
        res = eng.predict(text, imgs)
        got = np.array([np.log(res.probs[c] / (1 - res.probs[c])) for c in CATEGORY_KEYS])
        d = float(np.abs(ref - got).max())
        worst = max(worst, d)
        sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
        dec_ok = bool(((sig(ref) > 0.5) == (sig(got) > 0.5)).all())
        print(f"{name:16s} T={res.n_tokens:4d} max|dlogit|={d:.5f} dprob={np.abs(sig(ref) - sig(got)).max():.5f} "
              f"decisions_equal={dec_ok} total={res.timings_ms['total']:.0f}ms")
    print(f"WORST max|dlogit| = {worst:.5f}")
    if a.tol is not None and worst > a.tol:
        print("FAIL: превышен допуск")
        sys.exit(1)


if __name__ == "__main__":
    main()
