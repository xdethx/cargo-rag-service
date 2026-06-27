FROM python:3.11-slim
WORKDIR /app

# Default port; override with the PORT env var at runtime.
ENV PORT=7860

# Cache HF model weights under the app dir.
# HF Docker Spaces runs as a non-root user (uid 1000) whose $HOME is not
# writable, so the default ~/.cache path would fail at build or runtime.
ENV HF_HOME=/app/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Bake the embedding model into the image at build time.
# Reads the model name from src/config.py (EMBEDDING_MODEL env var or its
# default) — the model name is never hardcoded here.
# This avoids re-downloading weights on every ephemeral cold start and prevents
# cache-write errors under the non-root runtime user.
RUN python -c "from src.config import get_embeddings; get_embeddings()"

# The prebuilt FAISS index is committed to the repo; COPY it in after the
# heavyweight embedding step so index changes don't bust the model-download layer.
COPY data/index/ ./data/index/

# Make weights and index world-readable so the non-root runtime user can access them.
RUN chmod -R a+rX /app/.cache /app/data

EXPOSE 7860

CMD ["python", "-m", "src.main"]
