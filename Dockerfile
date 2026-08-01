# CPU inference container for archaeological hole detection.
#
# Base: Debian slim + Python 3.12 with the CPU-portable torch wheel from
# requirements.txt (no CUDA runtime). For GPU inference, build your own image
# on a CUDA runtime base and install requirements-gpu.txt first.
FROM python:3.12-slim

# Avoid interactive prompts and byte-compilation noise in the image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install core dependencies first (layers cached unless requirements change)
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application code and documentation
COPY configs/ configs/
COPY scripts/ scripts/
COPY docs/ docs/
COPY paper/ paper/

# Run as a non-root user
RUN useradd --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app

# Show inference CLI usage by default
CMD ["python", "scripts/inference.py", "--help"]
