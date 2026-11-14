"""Общее для инструментов: загрузка базы Qwen3.5, слияние LoRA-чекпоинта, голова классификатора."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

CATEGORY_KEYS = ["dangerous", "spam", "junk", "mismatch"]
N_CV_FEATURES = 9


def load_base(model_dir: str | Path):
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=torch.float32).eval()


def merge_lora_and_head(model, checkpoint_path: str | Path | None, hidden_size: int, seed: int = 0):
    """Вливает LoRA из `train_lora.py` checkpoint.pt в веса модели (in-place) и возвращает (W, b) головы.

    Формат чекпоинта (см. save_checkpoint в train_lora.py):
      lora_state: {"<module path>.lora_A": [r, in], "<module path>.lora_B": [out, r]}
      head_state: {"weight": [4, hidden + 9], "bias": [4]}
      lora_r, lora_alpha -> scale = alpha / r
    Если checkpoint_path is None — голова инициализируется случайно (режим отладки).
    """
    if checkpoint_path is None:
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(len(CATEGORY_KEYS), hidden_size + N_CV_FEATURES, generator=g) * 0.02
        b = torch.zeros(len(CATEGORY_KEYS))
        return w.numpy().astype(np.float32), b.numpy().astype(np.float32), {"random_head": True}

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    scale = ckpt["lora_alpha"] / ckpt["lora_r"]
    sd = ckpt["lora_state"]
    params = dict(model.named_parameters())
    merged = 0
    for key in sd:
        if not key.endswith(".lora_A"):
            continue
        prefix = key[: -len(".lora_A")]
        a = sd[key].float()
        bmat = sd[prefix + ".lora_B"].float()
        # prefix = путь к LoRALinear == путь исходного nn.Linear в обычной модели
        name = prefix + ".weight"
        if name not in params:
            raise KeyError(f"LoRA-цель {name} не найдена в базовой модели")
        with torch.no_grad():
            params[name].add_(scale * (bmat @ a))
        merged += 1
    head = ckpt["head_state"]
    w = head["weight"].float().numpy()
    b = head["bias"].float().numpy()
    assert w.shape == (len(CATEGORY_KEYS), hidden_size + N_CV_FEATURES), w.shape
    return w, b, {"random_head": False, "merged_lora_layers": merged, "lora_scale": scale}
