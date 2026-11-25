#!/usr/bin/env python3
"""Экспорт SmolVLM2-256M (full fine-tune, models/trained_model) + classification_head.pt в ONNX.

    uv run --group tools python -m tools.export_onnx_smolvlm --out artifacts/smolvlm_fp32

Модель дообучена целиком (train_vlm.py, без LoRA), архитектура стандартная: SigLIP vision + VLlama3
(обычный Llama с GQA). Экспорт переносит формулы из transformers напрямую (tools/wrappers_smolvlm.py).

Артефакты в --out (их же читают tools.quantize и service/app):
  vision.onnx(+.data)   pixel_values[N,3,image_size,image_size] -> image_embeds[N*tokens_per_image, hidden]
  lm.onnx(+.data)       inputs_embeds[1,T,hidden], cos/sin[T,head_dim] -> hidden[1,hidden] (последний токен)
  embed_tokens.npy      fp16 [vocab, hidden]
  head.npz              W[4, hidden+9], b[4]  (из classification_head.pt)
  meta.json             константы для препроцессинга и сборки промпта
  tokenizer.json        токенизатор (в trained_model своего нет — берётся из --tokenizer-dir)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import wrappers_smolvlm as W  # noqa: E402

CATEGORY_KEYS = ["dangerous", "spam", "junk", "mismatch"]
N_CV_FEATURES = 9


def load_head(path: Path, hidden_size: int) -> tuple[np.ndarray, np.ndarray]:
    sd = torch.load(path, map_location="cpu")
    w, b = sd["weight"].float().numpy(), sd["bias"].float().numpy()
    assert w.shape == (len(CATEGORY_KEYS), hidden_size + N_CV_FEATURES), w.shape
    return w, b


def export_all(model_dir: Path, tokenizer_dir: Path, out: Path, opset: int = 23) -> dict:
    from transformers import AutoModelForImageTextToText

    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=torch.float32).eval()
    cfg = model.config
    tcfg, vcfg = cfg.text_config, cfg.vision_config
    head_w, head_b = load_head(model_dir / "classification_head.pt", tcfg.hidden_size)

    text_model = model.model.text_model
    vision_model = model.model.vision_model
    connector = model.model.connector

    # ---- embeddings -> npy (fp16)
    emb = text_model.embed_tokens.weight.detach().to(torch.float16).numpy()
    np.save(out / "embed_tokens.npy", emb)
    np.savez(out / "head.npz", W=head_w, b=head_b)

    head_dim = getattr(tcfg, "head_dim", tcfg.hidden_size // tcfg.num_attention_heads)

    # ---- language model (prefill-only, без KV-кэша)
    lm = W.TextWrap(text_model).eval()
    T0 = 48
    ex_lm = (torch.randn(1, T0, tcfg.hidden_size) * 0.1, torch.randn(T0, head_dim), torch.randn(T0, head_dim))
    Tdim = torch.export.Dim("T", min=1, max=4096)
    ds_lm = ({1: Tdim}, {0: Tdim}, {0: Tdim})
    print("exporting lm ...")
    prog = torch.onnx.export(lm, ex_lm, dynamo=True, dynamic_shapes=ds_lm, opset_version=opset,
                             input_names=["inputs_embeds", "cos", "sin"], output_names=["hidden"])
    prog.save(str(out / "lm.onnx"), external_data=True)

    # ---- vision
    vw = W.VisionWrap(vision_model, connector).eval()
    image_size = vcfg.image_size
    ex_vis = (torch.randn(2, 3, image_size, image_size),)
    Ndim = torch.export.Dim("N", min=1, max=4)
    ds_vis = ({0: Ndim},)
    print("exporting vision ...")
    prog = torch.onnx.export(vw, ex_vis, dynamo=True, dynamic_shapes=ds_vis, opset_version=opset,
                             input_names=["pixel_values"], output_names=["image_embeds"])
    prog.save(str(out / "vision.onnx"), external_data=True)

    tokens_per_image = (vcfg.image_size // vcfg.patch_size) ** 2 // (cfg.scale_factor ** 2)
    inv_freq = (1.0 / (tcfg.rope_parameters["rope_theta"]
                       ** (torch.arange(0, head_dim, 2).float() / head_dim))).tolist()

    meta = {
        "hidden_size": tcfg.hidden_size,
        "vocab_size": tcfg.vocab_size,
        "image_token_id": cfg.image_token_id,
        "fake_image_token": "<fake_token_around_image>",
        "global_image_token": "<global-img>",
        "image_token": "<image>",
        "bos_token": "<|im_start|>",
        "end_of_utterance_token": "<end_of_utterance>",
        "image_size": image_size,
        "patch_size": vcfg.patch_size,
        "tokens_per_image": tokens_per_image,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "head_dim": head_dim,
        "rope_theta": tcfg.rope_parameters["rope_theta"],
        "text_inv_freq": inv_freq,
        "categories": CATEGORY_KEYS,
        "n_cv_features": N_CV_FEATURES,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.copy(tokenizer_dir / "tokenizer.json", out / "tokenizer.json")
    print("done ->", out)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/trained_model")
    ap.add_argument("--tokenizer-dir", default="models/SmolVLM2-256M",
                    help="в trained_model нет сохранённого токенизатора")
    ap.add_argument("--out", default="artifacts/smolvlm_fp32")
    ap.add_argument("--opset", type=int, default=23)
    a = ap.parse_args()
    export_all(Path(a.model_dir), Path(a.tokenizer_dir), Path(a.out), a.opset)


if __name__ == "__main__":
    main()
