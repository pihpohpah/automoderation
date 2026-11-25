# Пайплайн: чекпоинт → ONNX → квантование

Нужен, когда меняется чекпоинт (`models/trained_model`) или схема квантования. Результат, `artifacts/smolvlm_int8/`,
в git не хранится, его надо собрать или скопировать на машину перед `docker build`.

```bash
export UV_CACHE_DIR=$PWD/.uv-cache
uv sync --group tools --group dev

# 1) экспорт: веса из models/trained_model (full fine-tune, train_vlm.py), классификационная голова
#    из classification_head.pt. Токенизатор берётся из models/SmolVLM2-256M (в trained_model его нет)
uv run python -m tools.export_onnx_smolvlm --model-dir models/trained_model --out artifacts/smolvlm_fp32

# 2) квантование; режимы для lm и vision задаются раздельно (см. quantization.md)
uv run python -m tools.quantize --src artifacts/smolvlm_fp32 --dst artifacts/smolvlm_int8 \
  --mode "lm=w8,vision=int8" --embed int8                                    # рекомендовано (быстро, cos~0.88-0.998)
uv run python -m tools.quantize --src artifacts/smolvlm_fp32 --dst artifacts/smolvlm_w8 \
  --mode w8 --embed int8                                                     # точнее (cos~0.99+), vision ~2.3x медленнее

# 3) проверка паритета с PyTorch/HF и качества на val (--group tools --with num2words нужны для AutoProcessor)
uv run --with num2words python -m tools.verify_smolvlm --artifacts artifacts/smolvlm_fp32 --mode engine --tol 0.1
uv run python -m tools.eval_val --artifacts artifacts/smolvlm_int8 --ref artifacts/smolvlm_fp32 --dataset dataset --limit 300
```

## Что лежит в артефактах

```
artifacts/smolvlm_int8/{vision.onnx, vision.onnx.data, lm.onnx, lm.onnx.data,
                        embed_tokens.npy, embed_scale.npy, head.npz, meta.json, tokenizer.json}
```

- `vision.onnx`: SigLIP ViT и pixel-shuffle connector. Позиции задаёт learned position embedding, RoPE здесь нет.
- `lm.onnx`: 30 слоёв стандартного Llama-внимания с GQA. Возвращает hidden последнего токена после final norm.
- `embed_tokens.npy`: int8 с масштабом по строке (`embed_scale.npy`), сервис читает его через mmap.
- `head.npz`: веса классификационной головы.
- `meta.json`: константы для препроцессинга и сборки промпта.

Бэкбон обычный Llama, поэтому `tools/wrappers_smolvlm.py` переносит формулы из transformers напрямую: prefill
целиком, `is_causal=True` в SDPA, без KV-кэша.

## Инструменты

| Скрипт | Назначение |
|---|---|
| `tools/export_onnx_smolvlm.py`, `tools/wrappers_smolvlm.py` | экспорт чекпоинта в ONNX |
| `tools/quantize.py` | квантование (`int8`, `w8`, `q4`, раздельно по частям) |
| `tools/verify_smolvlm.py` | паритет ONNX ↔ PyTorch/HF (`--mode onnx` только графы, `--mode engine` весь `service/app`) |
| `tools/eval_val.py` | AUC/AP на `val.jsonl` и расхождение с fp32 (`--ref`) |
| `tools/compare_hidden.py` | cos(hidden) между fp32 и квантованными артефактами на синтетических кейсах |
| `tools/bench.py` | худший случай против запущенного сервиса + пиковый RSS |
| `tools/bench_smol.py` | скорость и память готовых HF ONNX-экспортов SmolVLM2-256M, без нашей головы |
