# Multi-stage build. Stage 1 installs wheels into a throwaway layer; stage 2
# copies just site-packages + source onto a slim runtime.
#
# Result: ~700 MB (vs ~1.4 GB single-stage) — most of the savings come from
# dropping build toolchain, pip cache, apt lists, and .pyc bytecode.

# stage 1: builder
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .

# Install into a prefix we can copy wholesale. Using --no-deps would be faster
# but breaks transitive deps; the multi-stage already trims the fat.
RUN pip install --prefix=/install -r requirements.txt \
 && find /install -name "__pycache__" -type d -exec rm -rf {} + \
 && find /install -name "*.pyc" -delete \
 && find /install -name "tests" -type d -prune -exec rm -rf {} + 2>/dev/null || true

# ---------- stage 2: runtime ----------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/        ./src/
COPY config.yaml .

EXPOSE 8501

CMD ["streamlit", "run", "src/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
