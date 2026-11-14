#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers>=4.57.0",
#   "torch",
#   "accelerate",
#   "pillow",
#   "numpy",
#   "scipy",
#   "scikit-learn",
#   "tqdm",
# ]
# ///
"""Полный (не LoRA) finetune SmolVLM-256M под мультилейбл-классификацию модерации (text+images).

Та же логика, что в train_lora.py (см. обсуждение в чате), адаптированная под маленькую
модель со стандартной архитектурой (Llama-бэкбон + SigLIP-энкодер, без экзотики вроде
гибридного linear attention в Qwen3.5 — поэтому не нужны particular CUDA-кернели и не
нужна LoRA, чтобы не повредить память: 256M целиком дообучается напрямую):
  - база (SmolVLMForConditionalGeneration) ПОЛНОСТЬЮ обучаема — в отличие от train_lora.py,
    тут нет замороженного визуального энкодера: модель достаточно маленькая, чтобы full
    finetune был дешевле и, вероятно, качественнее LoRA-адаптера поверх гиганта;
  - классификационная голова: Linear(hidden_size + n_cv_features -> 4) поверх hidden state
    последнего не-пад токена, обучается с нуля вместе с остальной моделью;
  - CV-признаки (блюр/яркость/контраст/энтропия) считаются на ОРИГИНАЛЬНОМ фото, до любого
    даунскейла под модель — та же причина, что в train_lora.py: ресайз смазывает именно тот
    сигнал, ради которого эти признаки существуют (детекция junk);
  - лосс: BCEWithLogitsLoss с pos_weight на класс (тот же дисбаланс dangerous/spam/junk/mismatch).

Модель лежит в репозитории локально (model_smolvlm256m/) — перенесите эту папку на офлайн-
сервер так же, как dataset.tar.gz и model/ для Qwen3.5.

Запуск:
    python3 train_vlm.py --model-name ./model_smolvlm256m --dataset-dir ./dataset \
        --device cuda --output-dir runs/vlm_v1
    python3 train_vlm.py --max-train-samples 8 --max-val-samples 4 --epochs 1 --batch-size 4  # smoke-test

LR по умолчанию (2e-5) НАМНОГО меньше, чем в train_lora.py (2e-4) — там обучались только
LoRA-адаптеры поверх замороженной сети, тут обновляются все предобученные веса напрямую,
большой LR их быстро разрушит (catastrophic forgetting).
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Снижает рост GPU-памяти из-за фрагментации кэширующего аллокатора (у нас батчи неоднородные
# по форме — 0-4 фото на запись — так что аллокатор периодически кэширует всё более крупные
# блоки). Должно быть выставлено до инициализации CUDA-контекста, поэтому до import torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import laplace
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CATEGORY_KEYS = ["dangerous", "spam", "junk", "mismatch"]
N_CV_FEATURES = 9  # [mean,min](log_blur, brightness, contrast, entropy) + n_images


# --------------------------------------------------------------------------- CV-признаки

def _image_cv_stats(img: Image.Image) -> np.ndarray:
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    blur = float(laplace(gray).var())
    log_blur = float(np.log1p(blur))
    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    hist, _ = np.histogram(gray, bins=64, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum()) if len(hist) else 0.0
    return np.array([log_blur, brightness, contrast, entropy], dtype=np.float32)


def compute_cv_features(images: list[Image.Image]) -> np.ndarray:
    if not images:
        return np.zeros(N_CV_FEATURES, dtype=np.float32)
    stats = np.stack([_image_cv_stats(img) for img in images])
    mean_stats = stats.mean(axis=0)
    min_stats = stats.min(axis=0)
    return np.concatenate([mean_stats, min_stats, [float(len(images))]]).astype(np.float32)


# --------------------------------------------------------------------------- Dataset

@dataclass
class Example:
    text: str
    images: list[Image.Image]
    labels: np.ndarray
    cv_features: np.ndarray


class ModerationDataset(Dataset):
    def __init__(self, jsonl_path: Path, max_images: int = 4, limit: int | None = None,
                 image_max_side: int | None = None):
        with open(jsonl_path, encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]
        if limit:
            self.records = self.records[:limit]
        self.dataset_dir = jsonl_path.parent
        self.max_images = max_images
        # SmolVLM сам ресайзит до max_image_size.longest_edge=512 внутри процессора —
        # этот внешний даунскейл опционален (экономит время декодирования больших исходников),
        # не обязателен для памяти, как было с Qwen3.5.
        self.image_max_side = image_max_side

    def __len__(self) -> int:
        return len(self.records)

    def _downscale(self, img: Image.Image) -> Image.Image:
        if self.image_max_side and max(img.size) > self.image_max_side:
            scale = self.image_max_side / max(img.size)
            img = img.resize(
                (max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS
            )
        return img

    def __getitem__(self, idx: int) -> Example:
        rec = self.records[idx]
        image_paths = rec["images"][: self.max_images]
        originals = [Image.open(self.dataset_dir / p).convert("RGB") for p in image_paths]
        cv_features = compute_cv_features(originals)
        images = [self._downscale(img) for img in originals]
        labels = np.array([rec["categories"][k] for k in CATEGORY_KEYS], dtype=np.float32)
        return Example(text=rec["text"] or "", images=images, labels=labels, cv_features=cv_features)


def build_collate_fn(processor):
    """SmolVLM/Idefics3Processor умеет текст+картинки одним вызовом apply_chat_template —
    в отличие от Qwen, не нужен отдельный шаг processor(text=..., images=...).
    do_image_splitting=False: тайлинг на под-изображения даёт точность на мелком тексте/OCR,
    не нужную для сцено-уровневой модерации, и резко раздувает число визуальных токенов
    при 2-4 фото на запись."""

    def collate(batch: list[Example]):
        # Картинки вшиваются прямо в content (а не передаются отдельным images=) —
        # при батче из НЕСКОЛЬКИХ диалогов apply_chat_template сам достаёт их из content
        # и падает с "got multiple values for keyword argument 'images'", если их ещё и
        # передать отдельным аргументом (это сработало бы только для одного диалога).
        texts_messages = []
        for ex in batch:
            content = [{"type": "image", "image": img} for img in ex.images] + [
                {"type": "text", "text": ex.text}
            ]
            texts_messages.append([{"role": "user", "content": content}])

        inputs = processor.apply_chat_template(
            texts_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            do_image_splitting=False,
        )
        labels = torch.tensor(np.stack([ex.labels for ex in batch]), dtype=torch.float32)
        cv_features = torch.tensor(np.stack([ex.cv_features for ex in batch]), dtype=torch.float32)
        return inputs, labels, cv_features

    return collate


# --------------------------------------------------------------------------- Модель

class ModerationClassifier(nn.Module):
    def __init__(self, base_model, hidden_size: int, n_cv_features: int = N_CV_FEATURES):
        super().__init__()
        self.base_model = base_model
        self.head = nn.Linear(hidden_size + n_cv_features, len(CATEGORY_KEYS))

    def _pooled_hidden(self, inputs) -> torch.Tensor:
        outputs = self.base_model(**inputs, output_hidden_states=True, use_cache=False)
        last_hidden = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"]
        last_token_idx = attention_mask.sum(dim=1) - 1  # верно при right-padding
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        return last_hidden[batch_idx, last_token_idx]

    def forward(self, inputs, cv_features: torch.Tensor) -> torch.Tensor:
        pooled = self._pooled_hidden(inputs)
        combined = torch.cat([pooled.float(), cv_features.float()], dim=-1)
        return self.head(combined)


def build_model(model_name: str, gradient_checkpointing: bool):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    # _pooled_hidden ищет последний не-пад токен через attention_mask.sum(-1)-1 — верно
    # только при right-padding (у SmolVLM это и так дефолт, но фиксируем явно).
    processor.tokenizer.padding_side = "right"

    base = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16)
    hidden_size = base.config.text_config.hidden_size

    # Полный finetune: ничего не замораживаем (в отличие от train_lora.py) — модель
    # достаточно маленькая, чтобы это было дешевле LoRA и адаптировало всю сеть целиком,
    # включая визуальный энкодер (там, в отличие от Qwen3.5+LoRA-MVP, мы не рискуем
    # недообучить junk/mismatch из-за заморозки — тут всё учится).
    for p in base.parameters():
        p.requires_grad = True

    if gradient_checkpointing:
        if hasattr(base, "activation_checkpointing_enable"):
            base.activation_checkpointing_enable()
        else:
            base.gradient_checkpointing_enable()

    classifier = ModerationClassifier(base, hidden_size)
    return classifier, processor


def save_checkpoint(model: ModerationClassifier, path: Path) -> None:
    """В отличие от train_lora.py (там сохранялись только LoRA A/B + голова), тут модель
    обучается целиком — сохраняем её как обычный transformers-чекпоинт (save_pretrained,
    грузится потом напрямую через AutoModelForImageTextToText.from_pretrained(path)),
    плюс отдельно голову, т.к. она не часть исходной архитектуры SmolVLM."""
    path.mkdir(parents=True, exist_ok=True)
    model.base_model.save_pretrained(path)
    torch.save(model.head.state_dict(), path / "classification_head.pt")


# --------------------------------------------------------------------------- Обучение

def compute_pos_weight(dataset: ModerationDataset) -> torch.Tensor:
    labels = np.stack([
        np.array([r["categories"][k] for k in CATEGORY_KEYS], dtype=np.float32)
        for r in dataset.records
    ])
    n_pos = labels.sum(axis=0)
    n_neg = len(labels) - n_pos
    return torch.tensor(n_neg / np.clip(n_pos, 1, None), dtype=torch.float32)


@torch.no_grad()
def evaluate(model: ModerationClassifier, loader: DataLoader, device: str) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    all_logits, all_labels = [], []
    for inputs, labels, cv_features in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(inputs, cv_features.to(device))
        all_logits.append(logits.cpu().float().numpy())
        all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1 / (1 + np.exp(-logits))

    metrics = {}
    for i, cat in enumerate(CATEGORY_KEYS):
        y_true, y_score = labels[:, i], probs[:, i]
        if len(np.unique(y_true)) < 2:
            metrics[cat] = {"auc": float("nan"), "ap": float("nan")}
            continue
        metrics[cat] = {
            "auc": float(roc_auc_score(y_true, y_score)),
            "ap": float(average_precision_score(y_true, y_score)),
        }
    overall_true = labels.max(axis=1)
    overall_score = probs.max(axis=1)
    if len(np.unique(overall_true)) == 2:
        metrics["overall"] = {"auc": float(roc_auc_score(overall_true, overall_score))}
    return metrics


def train(args: argparse.Namespace) -> None:
    device = args.device
    dataset_dir = Path(args.dataset_dir)

    train_ds = ModerationDataset(dataset_dir / "train.jsonl", args.max_images, args.max_train_samples,
                                  args.image_max_side)
    val_ds = ModerationDataset(dataset_dir / "val.jsonl", args.max_images, args.max_val_samples,
                                args.image_max_side)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    print("building model...")
    model, processor = build_model(args.model_name, args.gradient_checkpointing)
    print("model built, moving to device...")
    model.to(device)
    print("model on device")

    collate_fn = build_collate_fn(processor)
    # Модель крошечная (256M) — GPU почти наверняка будет ждать декодирование JPEG'ов и
    # сборку чат-шаблона на CPU, если это делать в главном процессе. num_workers>0 разносит
    # это по отдельным процессам, пока GPU считает предыдущий батч.
    loader_kwargs = dict(num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, **loader_kwargs)

    pos_weight = compute_pos_weight(train_ds).to(device)
    print("pos_weight per category:", dict(zip(CATEGORY_KEYS, pos_weight.tolist())))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%) — full finetune")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    accum = max(1, args.gradient_accumulation_steps)
    print(f"effective batch size: {args.batch_size * accum} (batch={args.batch_size} x accum={accum})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        progress = tqdm(train_loader, desc=f"epoch {epoch}", unit="step")
        for step, (inputs, labels, cv_features) in enumerate(progress):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            cv_features = cv_features.to(device)

            logits = model(inputs, cv_features)
            loss = criterion(logits, labels) / accum
            loss.backward()

            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accum
            progress.set_postfix(loss=f"{running_loss / (step + 1):.4f}")

        metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch} val metrics: {json.dumps(metrics, indent=2)}")

        save_checkpoint(model, output_dir / f"epoch_{epoch}")
        print(f"чекпоинт эпохи {epoch} сохранён в {output_dir / f'epoch_{epoch}'}")

    save_checkpoint(model, output_dir / "final")
    print(f"финальный чекпоинт сохранён в {output_dir / 'final'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="runs/vlm_v1")
    parser.add_argument("--model-name", default="./model_smolvlm256m",
                         help="HF id или локальный путь к папке с config.json/safetensors/tokenizer")
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument(
        "--image-max-side", type=int, default=None,
        help="Опциональный даунскейл перед подачей в процессор (сам SmolVLM и так ресайзит "
             "до max_image_size.longest_edge=512). 0/None — не даунскейлить.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Процессы DataLoader'а для декодирования фото/сборки чат-шаблона параллельно "
             "с GPU-вычислениями — модель маленькая, иначе CPU-препроцессинг станет узким "
             "местом вместо самой сети.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1,
        help="Эффективный batch = batch-size * это значение (без доп. GPU-памяти).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5,
        help="Full finetune — LR на порядок меньше, чем для LoRA (2e-4 в train_lora.py).",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
