# contentmoderration_service

Веб-сервис автомодерации постов (текст + 0..4 фото). Внутри — SmolVLM2-256M (Idefics3 vision + VLlama3
language model, полный fine-tune без LoRA — см. `train_vlm.py`) с классификационной головой,
экспортированный в ONNX, квантованный и запущенный на ONNX Runtime CPU. Цель: 1 CPU, ≤ 1 ГБ RAM, ответ ≤ ~10 с.

Категории модерации: `dangerous`, `spam`, `junk`, `mismatch` (текст не соответствует фото).

## Быстрый старт (Docker)

**Артефакты не лежат в git** (файлы `*.onnx.data` больше жёсткого лимита GitHub в 100 МБ — см. "Риски").
Перед сборкой положите `artifacts/smolvlm_int8/` руками (скопируйте с машины, где он уже собран, либо
соберите сами — см. "Пайплайн" ниже) так, чтобы получилось:

```
artifacts/smolvlm_int8/{vision.onnx, vision.onnx.data, lm.onnx, lm.onnx.data,
                        embed_tokens.npy, embed_scale.npy, head.npz, meta.json, tokenizer.json}
```

```bash
docker build -t moderation-service .
docker run --rm -p 8000:8000 --cpus=1 --memory=1g moderation-service
```

```bash
curl -F "text=Концерт в субботу, афиша во вложении" -F "images=@poster.jpg" http://localhost:8000/moderate
curl http://localhost:8000/health
```

## Быстрый старт (без Docker, для разработки)

```bash
export UV_CACHE_DIR=$PWD/.uv-cache
uv sync                                        # только рантайм-зависимости сервиса (fastapi/onnxruntime/...)
ARTIFACTS_DIR=artifacts/smolvlm_int8 uv run uvicorn app.main:app --app-dir service --port 8000
```

`uv sync` без флагов ставит только `[project.dependencies]` — лёгкий набор без torch/transformers.
Для работы с экспортом/квантованием/обучением (`tools/`, `train_vlm.py`) нужны доп. группы:

```bash
uv sync --group tools --group dev   # torch, transformers, onnx, pytest, scikit-learn, ...
```

## API

`POST /moderate` — `multipart/form-data`: `text` (строка), `images` (0..4 файлов, повторяющееся поле).

```bash
curl -F "text=Концерт в субботу, афиша во вложении" -F "images=@poster.jpg" http://localhost:8000/moderate
```

```json
{"label": 0, "flagged": [],
 "categories": {"dangerous": {"prob": 0.01, "flag": false}, "spam": {...}, "junk": {...}, "mismatch": {...}},
 "timings_ms": {"cv_features": 9, "vision": 230, "prompt": 0.4, "lm": 700, "total": 940, "queue_and_io": 4},
 "n_images": 1, "n_tokens": 149}
```

`label = 1`, если хотя бы одна категория выше порога (как в датасете). Пороги: `THR_DANGEROUS` (0.75) `|SPAM|JUNK|MISMATCH` (0.5).

`GET /health` — готовность + текущий/пиковый RSS процесса (для мониторинга укладывания в RAM-бюджет).

```json
{"status": "ok", "rss_mb": 350.1, "peak_rss_mb": 441.6}
```

Коды ошибок: `413` (текст > `MAX_TEXT_CHARS`, картинка > `MAX_IMAGE_BYTES` или слишком много пикселей),
`422` (> `MAX_IMAGES` фото, битая картинка), `503` + `Retry-After` (очередь переполнена — расчёт идёт строго
по одному запросу за раз), `504` (дедлайн `TIMEOUT_S` исчерпан; расчёт реально прерывается через
`RunOptions.terminate`, а не просто отбрасывается после таймаута — CPU не продолжает греться впустую).

### Настройки (переменные окружения)

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `ARTIFACTS_DIR` | `artifacts/smolvlm_int8` | каталог с ONNX-артефактами |
| `TIMEOUT_S` | `10` | общий бюджет на запрос (включая ожидание в очереди) |
| `MAX_IMAGES` | `4` | макс. число фото на запрос |
| `MAX_IMAGE_BYTES` | `8388608` (8 МБ) | макс. размер одного файла |
| `MAX_TEXT_CHARS` | `4000` | макс. длина текста в символах |
| `MAX_TEXT_TOKENS` | `512` | макс. длина текста в токенах (обрезается уже после лимита символов) |
| `MAX_QUEUE` | `2` | сколько запросов могут ждать своей очереди |
| `ORT_THREADS` | `1` | `intra_op_num_threads` ONNX Runtime — держите в `1` на 1 CPU |
| `WARMUP` | `1` | прогреть модель фиктивным запросом при старте |
| `THR_DANGEROUS` | `0.75` | порог `dangerous` (поднят с 0.5 — см. "Риски": модель путает `spam`/`dangerous`, на 0.5 было много ложных срабатываний) |
| `THR_SPAM` / `THR_JUNK` / `THR_MISMATCH` | `0.5` | пороги срабатывания по остальным категориям |

## Структура

```
service/app/        FastAPI-приложение: main.py (API, лимиты, дедлайн), engine.py (ORT), preprocess.py (numpy-препроцессинг)
tools/               export_onnx_smolvlm.py, wrappers_smolvlm.py, quantize.py, verify_smolvlm.py, eval_val.py, bench.py
                      (export_onnx.py/wrappers.py/common.py/verify.py — старый Qwen3.5-0.8B+LoRA пайплайн, не используется сервисом)
tests/               pytest
artifacts/           smolvlm_int8 (нужен для запуска) | smolvlm_fp32 | smolvlm_w8 — всё целиком не в git (см. .gitignore и "Риски")
models/              trained_model (веса, не в git) + SmolVLM2-256M (HF, источник токенизатора/конфига, не в git)
dataset/             train.jsonl/val.jsonl + images (не в git)
Dockerfile
pyproject.toml       project.dependencies — рантайм сервиса; dependency-groups tools/dev — экспорт/тесты
```

## Пайплайн: чекпоинт → ONNX → квантование

Нужен, только если меняется чекпоинт (`models/trained_model`) или схема квантования — для обычного запуска
сервиса `artifacts/smolvlm_int8` уже готов.

```bash
export UV_CACHE_DIR=$PWD/.uv-cache
uv sync --group tools --group dev

# 1) экспорт: веса из models/trained_model (full fine-tune, train_vlm.py), классификационная голова
#    из classification_head.pt. Токенизатор берётся из models/SmolVLM2-256M (в trained_model его нет)
uv run python -m tools.export_onnx_smolvlm --model-dir models/trained_model --out artifacts/smolvlm_fp32

# 2) квантование — lm и vision ведут себя по-разному под динамическим int8 (см. "Риски" ниже),
#    поэтому режимы задаются раздельно:
uv run python -m tools.quantize --src artifacts/smolvlm_fp32 --dst artifacts/smolvlm_int8 \
  --mode "lm=w8,vision=int8" --embed int8                                    # рекомендовано (быстро, cos~0.88-0.998)
uv run python -m tools.quantize --src artifacts/smolvlm_fp32 --dst artifacts/smolvlm_w8 \
  --mode w8 --embed int8                                                     # точнее (cos~0.99+), vision ~2.3x медленнее

# 3) проверка паритета с PyTorch/HF и качества на val (--group tools --with num2words нужны для AutoProcessor)
uv run --with num2words python -m tools.verify_smolvlm --artifacts artifacts/smolvlm_fp32 --mode engine --tol 0.1
uv run python -m tools.eval_val --artifacts artifacts/smolvlm_int8 --ref artifacts/smolvlm_fp32 --dataset dataset --limit 300
```

Что получается из SmolVLM2-256M: `vision.onnx` (SigLIP ViT + pixel-shuffle connector, без rope — только learned
position embedding), `lm.onnx` (30 слоёв стандартного Llama/GQA-внимания, возвращает hidden последнего токена
после final norm), `embed_tokens.npy` (int8 + масштаб по строке, mmap), `head.npz`. Архитектура — обычный
Llama-бэкбон (в отличие от Qwen3.5 без Gated DeltaNet/M-RoPE), поэтому экспорт — прямой перенос формул из
transformers (`tools/wrappers_smolvlm.py`), без chunked-трюков: prefill целиком, `is_causal=True` в SDPA,
без KV-кэша. `do_image_splitting=False` (как в `train_vlm.py`) — каждая картинка **безусловно** ресайзится
(1) аспект-сохраняюще так, чтобы длинная сторона стала ровно 2048 (а не "cap, если больше" — маленькие фото
тоже апскейлятся), затем (2) squash-ресайзом в 512×512 — это не no-op даже для уже маленьких фото, и его
пропуск был главным источником расхождения при разработке (см. "Риски").

## Результаты замеров (dev-контейнер, 4 CPU не throttled, intra_op=1, `smolvlm_int8`)

| Случай | Время | Пиковый RSS |
|---|---|---|
| текст короткий | 15-20 мс | 0.44 ГБ |
| текст 4000 симв. (517 токенов) | 0.48 с | 0.44 ГБ |
| 1 фото 1024px | 1.0 с | 0.46 ГБ |
| 4 фото 1024px + длинный текст | 4.8-5.0 с | 0.79 ГБ |
| 4 фото 4000px | 4.2-4.4 с | 0.81 ГБ |

Замеры не под `--cpus=1 --memory=1g` (нет Docker в окружении, где это разрабатывалось) — на целевом железе
латентность, скорее всего, выше; RSS от CPU-ограничения не зависит и должен быть достоверен. **Нужно
перепроверить под реальным 1 CPU/1 ГБ лимитом перед продом** (`docker run --cpus=1 --memory=1g` + `tools/bench.py`).

Паритет ONNX fp32 ↔ PyTorch/HF (сквозная проверка, numpy-препроцессинг сервиса): `max|Δlogit| ≈ 0.01-0.04` на
реальных фото из `val.jsonl`, 0 расхождений решений на 20 примерах. Квантование: `lm=w8,vision=int8` —
cos(hidden) 0.88-0.998 (2/15 расхождений решений на малой выборке, оба на 4-фото примерах); полный `w8` — cos 0.99+,
0 расхождений, но vision ~2.3x медленнее (9 с на 4 фото — слишком близко к дедлайну 10 с на медленном CPU).

AUC на 100 записях `val.jsonl` (`smolvlm_int8`): `dangerous` 0.86, `spam` 0.98, `junk` 1.00, `mismatch` 0.62,
**overall 0.85**. int8 vs fp32: decision flips@0.5 — 7%. `mismatch` заметно слабее остальных категорий — нужно
перепроверить на полном val.jsonl (1000 записей, здесь посчитано только 100 из-за времени расчёта).

## Что нужно знать / риски

* **`artifacts/` целиком не в git** — `lm.onnx.data` (~108 МБ) и `vision.onnx.data` (~94 МБ) больше жёсткого
  лимита GitHub в 100 МБ на файл (пуш с таким блобом отклоняется целиком, без исключений без Git LFS). Кладите
  `artifacts/smolvlm_int8/` на целевую машину руками (копией или пересборкой — см. "Пайплайн") перед
  `docker build`/запуском сервиса.
* **`THR_DANGEROUS` поднят с 0.5 до 0.75.** На дефолтном 0.5 модель давала много ложных срабатываний по
  `dangerous` (precision 0.69 на выборке из val.jsonl). Разбор ложноположительных показал: модель путает
  `spam` и `dangerous` на уровне признаков (38% FP — записи с лейблом только `spam`), это не баг сервиса и не
  артефакт квантования. На 0.75 precision 0.90 при том же recall, что и на 0.70 (чистый выигрыш — между 0.70 и
  0.75 в вероятностях сидит кластер безопасного контента). **Это калибровочная мера, не фикс первопричины** —
  если в проде вырастет доля спама, FPR по `dangerous` может снова подрасти; нормальное решение — донастройка
  с явным разделением spam/dangerous в данных.
* **Динамическое int8-квантование (`quantize_dynamic`) катастрофически ломает `lm.onnx` на этой модели** —
  cos(hidden) падал до ~0.70 (вместо ожидаемых 0.97+), даже на тексте без картинок. Похоже на известный эффект
  выбросов в активациях маленьких LLM, из-за которого on-the-fly квантование активаций работает плохо. Диагноз:
  изолированное квантование только `lm` (int8) давало тот же провал, только `embed_tokens` (int8) — cos 0.999
  (не виновник). Фикс — **`lm` квантуется в `w8`** (только веса, активации fp32); `vision` переживает динамический
  int8 нормально (cos 0.88-0.98). `tools/quantize.py --mode` принимает раздельные режимы (`lm=w8,vision=int8`),
  а не только один режим на весь экспорт. **Если будете трогать квантование — сначала прогоните
  `tools.eval_val --ref` и проверьте cos(hidden)/decision flips, не полагайтесь на то, что int8 "просто работает"
  для новых архитектур.**
* **Ресайз картинки — не "cap до 2048, если больше"**, а безусловный аспект-сохраняющий ресайз длинной стороны
  РОВНО до 2048 (`_resize_output_size_rescale_to_max_len` в HF), потом squash в 512×512. Пропуск первого шага для
  уже маленьких фото (казалось бы no-op) давал max|Δlogit| до 0.87 на реальных фото — тихая деградация качества,
  а не крэш. Если меняете `service/app/preprocess.py::resize_to_square` — обязательно гоняйте
  `tools.verify_smolvlm --mode engine` на реальных фото (не на синтетике), разница заметна только на фото, не на
  тексте.
* **Токенизация промпта должна идти ОДНИМ вызовом `encode()`** на собранную строку (границы не на спецтокенах) —
  раздельная токенизация кусков ломает BPE-слияние на границе (пробел перед кириллицей токенизировался иначе),
  давая `max|Δlogit| ≈ 0.13` даже на чистом тексте. См. `preprocess.py::build_prompt_string` +
  `engine.py::_encode_prompt`.
* **Качество (AUC) проверено только на 100/1000 записей val.jsonl** (полный прогон долго считается в
  single-threaded окружении) — прогнать `tools.eval_val` целиком перед продом, особенно для `mismatch` (800/10000
  записей в датасете — на малой выборке AUC может быть статистически шумным).
* Прод-сервис — один воркер; для масштабирования добавляйте реплики, а не потоки.
