#!/usr/bin/env python3
"""Квантование ONNX-артефактов (веса MatMul/Gemm; активации остаются fp32 — либо dynamic int8).

    python -m tools.quantize --src artifacts/smolvlm_fp32 --dst artifacts/smolvlm_int8 --mode lm=w8,vision=int8 --embed int8
    python -m tools.quantize --src artifacts/smolvlm_fp32 --dst artifacts/smolvlm_q4   --mode q4 --block-size 64

Режимы:
  int8  — onnxruntime quantize_dynamic: веса int8 (per-channel), активации квантуются на лету (uint8).
          Быстро на CPU (VNNI/NEON dotprod), но чувствительнее к выбросам активаций.
  w8    — только веса int8 (MatMulNBits 8 бит, блочный), активации fp32. Точнее int8-dynamic, чуть медленнее.
  q4    — только веса int4, блочные (MatMulNBits, block_size=64 по умолчанию, симметричное). Минимум памяти.

Также: embed_tokens.npy можно сжать до int8 с масштабом по строке (--embed int8) или оставить fp16 (default).
Любая часть (vision / lm) выбирается флагом --parts.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _quant_int8_dynamic(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(src), str(dst),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
        use_external_data_format=True,
        extra_options={"MatMulConstBOnly": True},
    )


def _quant_nbits(src: Path, dst: Path, bits: int, block_size: int) -> None:
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

    model = onnx.load(str(src), load_external_data=True)
    cfg = DefaultWeightOnlyQuantConfig(block_size=block_size, is_symmetric=True, bits=bits)
    q = MatMulNBitsQuantizer(model, bits=bits, block_size=block_size, is_symmetric=True, algo_config=cfg)
    q.process()
    q.model.save_model_to_file(str(dst), use_external_data_format=True)


def quantize_model(src: Path, dst: Path, mode: str, block_size: int) -> None:
    if dst.exists():
        dst.unlink()
    data = dst.with_name(dst.name + ".data")
    if data.exists():
        data.unlink()
    if mode == "int8":
        _quant_int8_dynamic(src, dst)
    elif mode == "w8":
        _quant_nbits(src, dst, 8, block_size)
    elif mode == "q4":
        _quant_nbits(src, dst, 4, block_size)
    else:
        raise ValueError(mode)


def quantize_embeddings_int8(src: Path, dst: Path) -> None:
    emb = np.load(src, mmap_mode="r")
    scale = np.zeros(emb.shape[0], np.float32)
    q = np.zeros(emb.shape, np.int8)
    for s in range(0, emb.shape[0], 16384):
        blk = np.asarray(emb[s:s + 16384], dtype=np.float32)
        sc = np.maximum(np.abs(blk).max(1), 1e-8) / 127.0
        q[s:s + 16384] = np.clip(np.round(blk / sc[:, None]), -127, 127).astype(np.int8)
        scale[s:s + 16384] = sc
    np.save(dst, q)
    np.save(dst.with_name("embed_scale.npy"), scale)


def _parse_mode(mode_arg: str, parts: list[str]) -> dict[str, str]:
    """--mode int8 -> один режим на все --parts; --mode lm=w8,vision=int8 -> раздельно по частям
    (динамический int8 на SmolVLM2 даёт cos(hidden) ~0.70 на lm и 0.88-0.98 на vision, w8 чинит lm;
    см. docs/quantization.md)."""
    if "=" not in mode_arg:
        return {p: mode_arg for p in parts}
    out = {}
    for chunk in mode_arg.split(","):
        name, mode = chunk.split("=")
        out[name.strip()] = mode.strip()
    missing = set(parts) - set(out)
    if missing:
        raise ValueError(f"--mode не задаёт режим для частей {missing}; укажи все --parts или используй один режим на все")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="artifacts/smolvlm_fp32")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--mode", default="int8",
                    help="int8|w8|q4 (один режим на все --parts) либо раздельно: lm=w8,vision=int8")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--parts", default="lm,vision", help="какие графы квантовать (через запятую)")
    ap.add_argument("--embed", choices=["fp16", "int8"], default="fp16")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    mode_by_part = _parse_mode(a.mode, parts)
    for name in ("lm", "vision"):
        s, d = src / f"{name}.onnx", dst / f"{name}.onnx"
        if name in parts:
            print(f"quantizing {name} ({mode_by_part[name]}) ...")
            quantize_model(s, d, mode_by_part[name], a.block_size)
        else:
            shutil.copy(s, d)
            shutil.copy(s.with_name(s.name + ".data"), d.with_name(d.name + ".data"))
    for f in ("head.npz", "meta.json", "tokenizer.json"):
        shutil.copy(src / f, dst / f)
    if a.embed == "int8":
        quantize_embeddings_int8(src / "embed_tokens.npy", dst / "embed_tokens.npy")
        meta = json.loads((dst / "meta.json").read_text())
        meta["embed_dtype"] = "int8"
        (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    else:
        shutil.copy(src / "embed_tokens.npy", dst / "embed_tokens.npy")
    total = sum(p.stat().st_size for p in dst.iterdir())
    print(f"done -> {dst}  ({total / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
