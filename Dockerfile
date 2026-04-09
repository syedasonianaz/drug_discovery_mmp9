# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.10 slim — matches conda env, keeps image lean
FROM python:3.10-slim

# ── System dependencies ────────────────────────────────────────────────────────
# libXrender and libXext required by RDKit for molecule rendering
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ────────────────────────────────────────────────
# Copy requirements first — Docker caches this layer
# Only reinstalls if requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────────────────────
# Models — needed at startup by DrugPredictor
COPY download_models.py .
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Source code
COPY src/ ./src/

# ── Environment ────────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces runs as non-root user
# This prevents permission errors on deployment
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# ── Port ───────────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Entry point ────────────────────────────────────────────────────────────────
CMD ["python", "src/app.py"]
