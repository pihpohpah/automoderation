#!/usr/bin/env python3
"""Экспорт Qwen3.5-0.8B (+ LoRA + голова) в набор ONNX-артефактов для CPU-сервиса.

    uv run --group tools python -m tools.export_onnx \
        --model-dir models/Qwen3.5-0.8B --checkpoint runs/lora_v1/checkpoint.pt --out artifacts/fp32

Без --checkpoint берётся случайная голова (для отладки пайплайна).

Артефакты в --out:
  vision.onnx(+.data)   pixel_values[N,1536], interp_indices[N,4], interp_weights[N,4], rot_cos/rot_sin[N,64] -> [N/4,1024]
  lm.onnx(+.data)       inputs_embeds[1,T,1024], cos/sin[T,64], last_index[1] -> hidden[1,1024]   (T кратно CHUNK)
  embed_tokens.npy      fp16 [vocab, 1024]  (грузится через mmap, в RAM попадают только нужные строки)
  head.npz              W[4, 1024+9], b[4]
  meta.json             константы для препроцессинга (rope inv_freq, id токенов, размеры, ...)
  tokenizer.json        токенизатор
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
from tools import wrappers as W  # noqa: E402
from tools.common import CATEGORY_KEYS, N_CV_FEATURES, load_base, merge_lora_and_head  # noqa: E402


def export_all(model_dir: Path, checkpoint: Path | None, out: Path, opset: int = 23) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    model = load_base(model_dir)
    cfg = model.config
    tcfg, vcfg = cfg.text_config, cfg.vision_config

    head_w, head_b, merge_info = merge_lora_and_head(model, checkpoint, tcfg.hidden_size)
    print("merge:", merge_info)

    lm_inner = model.model.language_model
    vis_inner = model.model.visual

    # ---- embeddings -> npy (fp16)
    emb = lm_inner.embed_tokens.weight.detach().to(torch.float16).numpy()
    np.save(out / "embed_tokens.npy", emb)
    np.savez(out / "head.npz", W=head_w, b=head_b)

    # ---- language model
    lm = W.LanguageModelWrap(lm_inner).eval()
    T0 = 4 * W.CHUNK
    rot_dim = int(tcfg.head_dim * tcfg.rope_parameters.get("partial_rotary_factor", 1.0))
    ex = (torch.randn(1, T0, tcfg.hidden_size) * 0.1, torch.randn(T0, rot_dim), torch.randn(T0, rot_dim),
          torch.tensor([T0 - 5]))
    C = torch.export.Dim("C", min=1, max=64)
    ds = ({1: W.CHUNK * C}, {0: W.CHUNK * C}, {0: W.CHUNK * C}, None)
    print("exporting lm ...")
    prog = torch.onnx.export(lm, ex, dynamo=True, dynamic_shapes=ds, opset_version=opset,
                             input_names=["inputs_embeds", "cos", "sin", "last_index"], output_names=["hidden"])
    prog.save(str(out / "lm.onnx"), external_data=True)

    # ---- vision
    vw = W.VisionWrap(vis_inner).eval()
    hd = vcfg.hidden_size // vcfg.num_heads
    N0 = 64
    ex = (torch.randn(N0, 3 * vcfg.temporal_patch_size * vcfg.patch_size ** 2),
          torch.randint(0, vcfg.num_position_embeddings, (N0, 4)), torch.rand(N0, 4),
          torch.randn(N0, hd), torch.randn(N0, hd))
    Mdim = torch.export.Dim("M", min=1, max=4096)
    ds = ({0: 4 * Mdim}, {0: 4 * Mdim}, {0: 4 * Mdim}, {0: 4 * Mdim}, {0: 4 * Mdim})
    print("exporting vision ...")
    prog = torch.onnx.export(vw, ex, dynamo=True, dynamic_shapes=ds, opset_version=opset,
                             input_names=["pixel_values", "interp_indices", "interp_weights", "rot_cos", "rot_sin"],
                             output_names=["image_embeds"])
    prog.save(str(out / "vision.onnx"), external_data=True)

    meta = {
        "chunk": W.CHUNK,
        "hidden_size": tcfg.hidden_size,
        "vocab_size": tcfg.vocab_size,
        "image_token_id": cfg.image_token_id,
        "vision_start_token_id": cfg.vision_start_token_id,
        "vision_end_token_id": cfg.vision_end_token_id,
        "pad_token_id": 248044,
        "patch_size": vcfg.patch_size,
        "merge_size": vcfg.spatial_merge_size,
        "temporal_patch_size": vcfg.temporal_patch_size,
        "num_position_embeddings": vcfg.num_position_embeddings,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "mrope_section": list(tcfg.rope_parameters.get("mrope_section", [11, 11, 10])),
        "text_inv_freq": lm_inner.rotary_emb.inv_freq.float().tolist(),
        "vision_inv_freq": vis_inner.rotary_pos_emb.inv_freq.float().tolist(),
        "categories": CATEGORY_KEYS,
        "n_cv_features": N_CV_FEATURES,
        "merge_info": merge_info,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.copy(model_dir / "tokenizer.json", out / "tokenizer.json")
    print("done ->", out)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3.5-0.8B")
    ap.add_argument("--checkpoint", default=None, help="checkpoint.pt из train_lora.py; без него — случайная голова")
    ap.add_argument("--out", default="artifacts/fp32")
    ap.add_argument("--opset", type=int, default=23)
    ap.add_argument("--no-sdpa", action="store_true", help="явный softmax-attention вместо SDPA (opset 18 хватает; больше памяти)")
    a = ap.parse_args()
    W.USE_SDPA = not a.no_sdpa
    export_all(Path(a.model_dir), Path(a.checkpoint) if a.checkpoint else None, Path(a.out), a.opset)


if __name__ == "__main__":
    main()
