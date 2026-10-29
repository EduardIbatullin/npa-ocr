# Точная версия Python и Debian (trixie — Tesseract 5.5.0): плавающий 3.12-slim при сборке
# с нуля мог бы сменить Debian, а с ним и Tesseract.
FROM python:3.12.15-slim-trixie

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
    PADDLE_PDX_CACHE_HOME=/app/cache/paddlex \
    # Xet (новый бэкенд загрузки с Hugging Face) на практике давал систематические
    # обрывы соединения на больших файлах моделей (десятки минут, повторные
    # попытки) — classic HTTP-путь huggingface_hub надёжнее в этом окружении.
    HF_HUB_DISABLE_XET=1 \
    # PaddleX перед использованием даже уже закэшированных моделей по умолчанию
    # проверяет связь с хостером моделей ("Checking connectivity to the model
    # hosters, this may take a while") — при нестабильной сети сама эта проверка
    # зависает на минуты, хотя скачивать ничего не нужно. Модели уже в образе
    # (см. ниже preload_models.py) — проверка не нужна вообще.
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

WORKDIR /app

# PyTorch CPU-сборка (иначе pip тянет CUDA-колёса в несколько гигабайт).
# torch и torchvision ставятся одной командой из одного индекса: torchvision из PyPI
# не совместим с CPU-сборкой torch (ошибка "operator torchvision::nms does not exist")
# Версии всех пакетов — из constraints.txt (проверенный набор): иначе сборка с нуля тянет
# свежие версии, и образ может не собраться или распознавать иначе. PyPI — запасной индекс
# для зависимостей torch, закреплённых версий которых нет в индексе PyTorch; сами torch
# и torchvision «+cpu» есть только в индексе PyTorch.
COPY constraints.txt ./
RUN pip install --no-cache-dir -c constraints.txt torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# Зависимости отдельно — слой кэшируется при неизменном pyproject.toml.
# server — HTTP API (npa_ocr.server), см. README, «HTTP API».
# Ставятся до копирования кода, по заглушкам пакета и README: иначе любая правка src или
# README заново загружала бы все зависимости (при медленной сети — больше часа). Установка
# editable — пакет читается из /app/src, настоящий код ложится туда ниже.
COPY pyproject.toml ./
RUN mkdir -p src/npa_ocr && touch src/npa_ocr/__init__.py README.md \
    && pip install --no-cache-dir -c constraints.txt -e ".[models,server]"
COPY README.md ./
COPY src ./src

# Модели качаются и кэшируются здесь, на этапе сборки, а не при первом реальном
# запросе — см. docstring scripts/preload_models.py. После правки кода повторяется
# (~2 мин): скрипт берёт движки из npa_ocr.
COPY scripts ./scripts
RUN python scripts/preload_models.py

EXPOSE 8000

# По умолчанию — CLI (см. README, «Командная строка»): PDF и результат передаются томом.
#   docker run --rm -v "$PWD":/data npa-ocr /data/scan.pdf --out /data/out.docx
# Сервер (HTTP API, см. README, «HTTP API») — тот же образ, другой вход:
#   docker run --rm -p 8000:8000 --entrypoint uvicorn npa-ocr \
#       npa_ocr.server:app --host 0.0.0.0 --port 8000
ENTRYPOINT ["npa-ocr"]
