"""Чистый numpy/PIL-препроцессинг (без torch): зеркало HF-процессора SmolVLM2 (Idefics3) и train_vlm.py.

Что здесь делается:
  * resize картинки ровно в image_size x image_size (do_image_splitting=False в train_vlm.py —
    без тайлинга на под-изображения, без паддинга: каждая картинка занимает весь патч-грид,
    поэтому в отличие от Qwen3.5-VL тут не нужна ни bucketize-интерполяция позиционных
    эмбеддингов, ни attention-маска для vision-энкодера);
  * нормализация (mean=std=0.5, как у SmolVLMImageProcessor);
  * одномерный RoPE (без M-RoPE — image-токены получают обычные последовательные позиции);
  * 9 CV-признаков (должны совпадать с compute_cv_features из train_vlm.py).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- изображения


def resize_to_square(img: Image.Image, image_size: int = 512, longest_edge_cap: int = 2048) -> Image.Image:
    """Как SmolVLMImageProcessor: (1) безусловный аспект-сохраняющий ресайз так, чтобы длинная
    сторона стала РОВНО longest_edge_cap (это не "cap если больше" — маленькие фото тоже
    апскейлятся; см. _resize_output_size_rescale_to_max_len), затем (2) do_image_splitting=False
    -> squash-ресайз в image_size x image_size (без сохранения аспекта). Оба шага — LANCZOS."""
    img = img.convert("RGB")
    w, h = img.size
    aspect = w / h
    if w >= h:
        new_w = longest_edge_cap
        new_h = int(new_w / aspect)  # усечение, как в HF (не round)
        if new_h % 2 != 0:
            new_h += 1
    else:
        new_h = longest_edge_cap
        new_w = int(new_h * aspect)
        if new_w % 2 != 0:
            new_w += 1
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return img.resize((image_size, image_size), Image.LANCZOS)


def image_to_pixel_values(img: Image.Image, image_size: int = 512,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> np.ndarray:
    """-> [3, image_size, image_size] float32, нормализовано."""
    img = resize_to_square(img, image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0                       # [H,W,3]
    arr = (arr - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))                   # [3,H,W]


# --------------------------------------------------------------------------- промпт (как apply_chat_template)

def build_prompt_string(text: str, n_images: int, tokens_per_image: int) -> str:
    """Строка промпта как у SmolVLMProcessor.apply_chat_template(do_image_splitting=False) для
    одного сообщения role=user (без add_generation_prompt). Токенизировать нужно ОДНИМ вызовом
    encode() на весь результат: граница между "User:"/"User: " и текстом пользователя проходит по
    обычным (не спец-) символам, и BPE должен иметь возможность слить, например, пробел с первым
    байтом кириллицы — раздельная токенизация кусков даёт другие id и портит hidden state."""
    if n_images:
        parts = ["<|im_start|>User:"]
        for _ in range(n_images):
            parts.append("<fake_token_around_image><global-img>" + "<image>" * tokens_per_image
                         + "<fake_token_around_image>")
        parts.append(text or "")
    else:
        parts = ["<|im_start|>User: ", text or ""]
    parts.append("<end_of_utterance>\n")
    return "".join(parts)


# --------------------------------------------------------------------------- одномерный RoPE

def text_rope(n_tokens: int, inv_freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """position_ids = arange(T) (без M-RoPE — картиночные токены позиционируются как обычные)."""
    pos = np.arange(n_tokens, dtype=np.float32)
    freqs = pos[:, None] * inv_freq[None, :]                               # [T,F]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


# --------------------------------------------------------------------------- CV-признаки (как в train_vlm.py)

N_CV_FEATURES = 9


def _laplace_var(gray: np.ndarray) -> float:
    from scipy.ndimage import laplace
    return float(laplace(gray).var())


def image_cv_stats(img: Image.Image) -> np.ndarray:
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    log_blur = float(np.log1p(_laplace_var(gray)))
    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    hist, _ = np.histogram(gray, bins=64, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum()) if len(hist) else 0.0
    return np.array([log_blur, brightness, contrast, entropy], dtype=np.float32)


def compute_cv_features(images: list[Image.Image]) -> np.ndarray:
    if not images:
        return np.zeros(N_CV_FEATURES, dtype=np.float32)
    stats = np.stack([image_cv_stats(i) for i in images])
    return np.concatenate([stats.mean(0), stats.min(0), [float(len(images))]]).astype(np.float32)
