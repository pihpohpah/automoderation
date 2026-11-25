# contentmoderation_service

Веб-сервис автомодерации постов: текст и до четырёх фото. Модель SmolVLM2-256M (Idefics3 vision + VLlama3) дообучена
целиком в `train_vlm.py`, без LoRA, и дополнена классификационной головой. Она экспортирована в ONNX, квантована и
работает на ONNX Runtime CPU. Бюджет: 1 CPU, до 1 ГБ RAM, ответ до ~10 с.

Категории: `dangerous`, `spam`, `junk`, `mismatch` (текст не соответствует фото).

## Быстрый старт (Docker)

**Артефактов модели нет в git:** файлы `*.onnx.data` (`lm` ~108 МБ, `vision` ~94 МБ) больше лимита GitHub в 100 МБ
на файл. Перед сборкой положите `artifacts/smolvlm_int8/` с машины, где он уже собран, или соберите его по
[docs/pipeline.md](docs/pipeline.md).

```bash
docker build -t moderation-service .
docker run --rm -p 8000:8000 --cpus=1 --memory=1g moderation-service
```

## Быстрый старт (без Docker, для разработки)

```bash
export UV_CACHE_DIR=$PWD/.uv-cache
uv sync                                        # только рантайм-зависимости сервиса (fastapi/onnxruntime/...)
ARTIFACTS_DIR=artifacts/smolvlm_int8 uv run uvicorn app.main:app --app-dir service --port 8000
```

`uv sync` без флагов ставит только `[project.dependencies]`, без torch и transformers. Для экспорта, квантования,
обучения и тестов нужны группы `tools` и `dev`:

```bash
uv sync --group tools --group dev   # torch, transformers, onnx, pytest, scikit-learn, ...
```

## API

`POST /moderate`, `multipart/form-data`: `text` (строка) и `images` (0..4 файлов, повторяющееся поле).

```bash
curl -F "text=Концерт в субботу, афиша во вложении" -F "images=@poster.jpg" http://localhost:8000/moderate
```

```json
{"label": 0, "flagged": [],
 "categories": {"dangerous": {"prob": 0.01, "flag": false}, "spam": {...}, "junk": {...}, "mismatch": {...}},
 "timings_ms": {"cv_features": 9, "vision": 230, "prompt": 0.4, "lm": 700, "total": 940, "queue_and_io": 4},
 "n_images": 1, "n_tokens": 149}
```

`label = 1`, если хотя бы одна категория не ниже своего порога (как в датасете).

`GET /health` отвечает, когда модель загружена и прогрета, и отдаёт текущий и пиковый RSS процесса для мониторинга
RAM-бюджета.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "rss_mb": 350.1, "peak_rss_mb": 441.6}
```

Ошибки:

- `413`: текст длиннее `MAX_TEXT_CHARS`, файл больше `MAX_IMAGE_BYTES` или в картинке слишком много пикселей.
- `422`: фото больше `MAX_IMAGES` или картинка битая.
- `503` с `Retry-After`: очередь заполнена. Запросы считаются строго по одному.
- `504`: истёк `TIMEOUT_S`. Расчёт останавливается через `RunOptions.terminate`, и CPU сразу освобождается.

### Переменные окружения

Пример со всеми переменными: `.env.example`.

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `ARTIFACTS_DIR` | `artifacts/smolvlm_int8` | каталог с ONNX-артефактами |
| `TIMEOUT_S` | `10` | общий бюджет на запрос, включая ожидание в очереди |
| `SAFETY_MARGIN_S` | `0.4` | запас на сериализацию ответа и сеть, вычитается из `TIMEOUT_S` |
| `MAX_IMAGES` | `4` | макс. число фото на запрос |
| `MAX_IMAGE_BYTES` | `8388608` (8 МБ) | макс. размер одного файла |
| `MAX_TEXT_CHARS` | `4000` | макс. длина текста в символах |
| `MAX_TEXT_TOKENS` | `512` | макс. длина текста в токенах, обрезается после проверки символов |
| `MAX_QUEUE` | `2` | сколько запросов могут ждать своей очереди |
| `ORT_THREADS` | `1` | `intra_op_num_threads` ONNX Runtime, на 1 CPU держите `1` |
| `WARMUP` | `1` | прогреть модель фиктивным запросом при старте |
| `THR_DANGEROUS` | `0.75` | порог `dangerous`, почему не 0.5: [docs/thresholds.md](docs/thresholds.md) |
| `THR_SPAM` / `THR_JUNK` / `THR_MISMATCH` | `0.5` | пороги остальных категорий |

Сервис работает в одном воркере. Для масштабирования добавляйте реплики, а не потоки.

## Структура

```
service/app/        FastAPI-приложение: main.py (API, лимиты, дедлайн), engine.py (ORT), preprocess.py (numpy-препроцессинг)
tools/               экспорт в ONNX, квантование, проверка паритета, оценка качества, бенчмарки (см. docs/pipeline.md)
tests/               pytest
docs/                устройство пайплайна, квантования, препроцессинга, порогов и замеры
train_vlm.py         обучение
artifacts/           smolvlm_int8 (нужен для запуска) | smolvlm_fp32 | smolvlm_w8, не в git
models/              trained_model (веса) + SmolVLM2-256M (HF, источник токенизатора и конфига), не в git
dataset/             train.jsonl/val.jsonl + images, не в git
Dockerfile
pyproject.toml       project.dependencies: рантайм сервиса; dependency-groups tools/dev: экспорт и тесты
```

## Документация

- [docs/pipeline.md](docs/pipeline.md): чекпоинт → ONNX → квантование, состав артефактов, инструменты.
- [docs/quantization.md](docs/quantization.md): почему `lm=w8,vision=int8` и как проверять изменения.
- [docs/preprocessing.md](docs/preprocessing.md): ресайз, токенизация промпта, CV-признаки. Прочитайте до правок в `preprocess.py`.
- [docs/thresholds.md](docs/thresholds.md): пороги категорий.
- [docs/benchmarks.md](docs/benchmarks.md): латентность, RSS, паритет с PyTorch, AUC.

## TODO

- Перемерить латентность под `docker run --cpus=1 --memory=1g` и `tools/bench.py`: текущие замеры сделаны на 4 CPU без лимитов.
- Прогнать `tools.eval_val` на всём `val.jsonl` (1000 записей), особенно ради `mismatch`: сейчас AUC посчитан на 100.
- Дообучить модель с явным разделением spam/dangerous в данных. Порог `THR_DANGEROUS=0.75` лечит только симптом.
