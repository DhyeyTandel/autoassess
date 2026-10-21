# syntax=docker/dockerfile:1
FROM python:3.12-slim

# opencv / ultralytics runtime libs (opencv-python links libGL and glib).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# CPU-only torch. uv.lock resolves torch to the default (CUDA) Linux wheels,
# which would add ~2.5 GB of nvidia-* packages. Strategy:
#   1. install torch/torchvision from the PyTorch CPU index (+cpu wheels);
#   2. export the locked third-party deps (keeps uv.lock pins) minus torch,
#      torchvision, nvidia-* and triton, and install them; torch is already
#      present so nothing pulls the CUDA build back in;
#   3. install the project itself with --no-deps.
# Step 1 fails the build if the result is not a +cpu build.
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN uv pip install --system --no-cache \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        torch torchvision \
    && uv export --frozen --no-hashes --no-dev --no-emit-project \
        --no-emit-package torch --no-emit-package torchvision \
        -o /tmp/requirements.txt \
    && grep -v -E '^(nvidia-|triton)' /tmp/requirements.txt > /tmp/requirements-cpu.txt \
    && uv pip install --system --no-cache -r /tmp/requirements-cpu.txt \
    && uv pip install --system --no-cache --no-deps . \
    && rm -f /tmp/requirements*.txt \
    && python -c "import torch; assert '+cpu' in torch.__version__ and not torch.cuda.is_available(), torch.__version__"

COPY configs/ configs/
COPY app.py ./

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /tmp/Ultralytics \
    && chown -R app:app /app /tmp/Ultralytics
USER app

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    AUTOASSESS_DEVICE=cpu \
    GRADIO_MAX_FILE_SIZE=15mb \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)"]

CMD ["python", "app.py"]
