FROM python:3.12-slim

# Tesseract с русским языком; libgomp/libgl — для PaddlePaddle и OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-rus \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    FLAGS_use_mkldnn=false \
    HF_HOME=/app/cache/huggingface \
    PADDLE_PDX_CACHE_HOME=/app/cache/paddlex

WORKDIR /app

# PyTorch CPU-сборка (иначе pip тянет CUDA-колёса в несколько гигабайт).
# torch и torchvision ставятся одной командой из одного индекса: torchvision из PyPI
# не совместим с CPU-сборкой torch (ошибка "operator torchvision::nms does not exist")
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Зависимости отдельно — слой кэшируется при неизменном pyproject.toml
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[models]"

RUN mkdir -p cache

# CLI-образ, не сервер: вход — сам npa-ocr, PDF и результат передаются томом.
# Пример: docker run --rm -v "$PWD":/data npa-ocr /data/scan.pdf --out /data/out.docx
ENTRYPOINT ["npa-ocr"]
