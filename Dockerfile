FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    ARTIFACTS_DIR=/app/artifacts ORT_THREADS=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv UV_NO_CACHE=1
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /usr/local/bin/
WORKDIR /app
# манифест отдельным слоем — пересборка зависимостей только при его изменении, не при правке app/
COPY pyproject.toml uv.lock ./
# без --group tools/--group dev: это torch/transformers/pytest и т.п., сервису они не нужны
RUN uv sync --frozen --no-dev
COPY service/app ./app
# квантованные артефакты (результат tools.export_onnx_smolvlm + tools.quantize), положите в artifacts/ перед сборкой
COPY artifacts/smolvlm_int8 ./artifacts
ENV PATH="/app/.venv/bin:${PATH}"
EXPOSE 8000
# один воркер: модель одна, 1 CPU. Параллелизм обеспечивает очередь внутри приложения.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "15"]
