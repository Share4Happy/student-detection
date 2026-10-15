# ==============================================================================
# Student Detection API - Production Dockerfile (Astral uv Best Practices)
# Based on: https://docs.astral.sh/uv/guides/integration/docker/
# ==============================================================================

# Stage 1: Build virtual environment
FROM python:3.11-slim AS builder

# Install uv from official distroless image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Use system python, compile bytecode, and use copy mode for files
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Step 1: Install external dependencies using cache mount and bind mounts
# This layer is cached as long as uv.lock and pyproject.toml do not change
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --no-editable

# Step 2: Copy the project source files
COPY . /app

# Step 3: Install the project itself in non-editable mode
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ==============================================================================
# Stage 2: Final minimal runtime image
# ==============================================================================
FROM python:3.11-slim

# Install system runtime dependencies (OpenCV GL/GLib runtime, OpenMP, and curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the complete pre-built virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create mount points for model checkpoints and Hugging Face cache
RUN mkdir -p /app/data/models /app/data/samples /root/.cache/huggingface

EXPOSE 8080

# Health check verifies the API documentation endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8080/docs || exit 1

# Start the Uvicorn application server
CMD ["uvicorn", "student_detection.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
