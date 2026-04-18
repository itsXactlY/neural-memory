# ============================================================
# SmolVM Dockerfile — Neural Memory PoC (JackrabbitDLM)
# Multi-stage build for minimal image size
# ============================================================

# NOTE: Build from project root:
#   cd ~/projects/neural-memory-adapter
#   docker build -f smolvm/Dockerfile -t nm-poc .

# ----------------------------------------------------------
# Stage 1: Base — Python 3.12 on Alpine
# ----------------------------------------------------------
FROM python:3.12-alpine AS base

RUN apk add --no-cache \
    build-base \
    unixodbc-dev \
    libstdc++ \
    libgcc \
    curl \
    bash \
    sqlite \
    && rm -rf /var/cache/apk/*

WORKDIR /app

# ----------------------------------------------------------
# Stage 2: Dependencies — install Python packages
# ----------------------------------------------------------
FROM base AS deps

COPY smolvm/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --prefix=/install -r /tmp/requirements.txt

# ----------------------------------------------------------
# Stage 3: Native libs — copy libneural_memory.so
# ----------------------------------------------------------
FROM base AS native

COPY build/libneural_memory.so /tmp/libneural_memory.so

# ----------------------------------------------------------
# Stage 4: Python source — neural memory module
# ----------------------------------------------------------
FROM base AS python-src

COPY python/ /tmp/python/
COPY __init__.py /tmp/__init__.py

# ----------------------------------------------------------
# Stage 5: Embed model placeholder (downloaded on first run)
# ----------------------------------------------------------
FROM base AS model

RUN mkdir -p /app/models/bge-m3

# ----------------------------------------------------------
# Stage 6: DLM bridge
# ----------------------------------------------------------
FROM base AS dlm-bridge

COPY smolvm/dlm_mssql_bridge.py /tmp/dlm_mssql_bridge.py

# ----------------------------------------------------------
# Stage 7: Final — assemble everything
# ----------------------------------------------------------
FROM base AS final

# Copy Python deps from stage 2
COPY --from=deps /install/lib/python3.12/site-packages /usr/lib/python3.12/site-packages
COPY --from=deps /install/bin /usr/local/bin

# Copy native lib from stage 3
COPY --from=native /tmp/libneural_memory.so /app/libneural_memory.so

# Copy Python source from stage 4
COPY --from=python-src /tmp/python/ /app/python/
COPY --from=python-src /tmp/__init__.py /app/__init__.py

# Copy model dir from stage 5
COPY --from=model /app/models /app/models

# Copy DLM bridge from stage 6
COPY --from=dlm-bridge /tmp/dlm_mssql_bridge.py /app/dlm_mssql_bridge.py

# Copy init scripts and configs
COPY smolvm/init.sh /app/init.sh
COPY smolvm/start-embed-server.sh /app/start-embed-server.sh
COPY smolvm/start-dlm-client.sh /app/start-dlm-client.sh
COPY smolvm/Smolfile /app/Smolfile
COPY smolvm/odbcinst.ini /etc/odbcinst.ini

RUN chmod +x /app/init.sh /app/start-embed-server.sh /app/start-dlm-client.sh

# Create data directory
RUN mkdir -p /app/data /app/logs

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    SMOLVM_DATA_DIR=/app/data \
    SMOLVM_LOG_DIR=/app/logs \
    SMOLVM_MODEL_DIR=/app/models/bge-m3 \
    SMOLVM_EMBED_PORT=8501 \
    SMOLVM_SQLITE_PATH=/app/data/neural_memory.db \
    DLM_HOST=host.docker.internal \
    DLM_PORT=37373 \
    DLM_TIMEOUT=10 \
    ODBCSYSINI=/etc

EXPOSE 8501

VOLUME ["/app/data", "/app/models"]

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8501/health || exit 1

ENTRYPOINT ["/app/init.sh"]
