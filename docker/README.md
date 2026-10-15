# Docker Deployment Guide

This directory contains the Docker configuration for deploying the Student Detection API service with GPU acceleration.

---

## 1. Prerequisites

- **Docker Engine**: Version $\ge 24.0$
- **Docker Compose**: Version $\ge 2.20$
- **NVIDIA GPU Drivers**: Installed on host system
- **NVIDIA Container Toolkit**: Installed and configured for Docker runtime

Verify NVIDIA Container Toolkit on your host:
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

---

## 2. Quick Start with Docker Compose (Recommended)

From the project root directory, run:

### Build and Start Container in Background
```bash
docker compose -f docker/docker-compose.yml up --build -d
```

### View Live Container Logs
```bash
docker compose -f docker/docker-compose.yml logs -f
```

### Stop Container Service
```bash
docker compose -f docker/docker-compose.yml down
```

---

## 3. Quick Start with Plain Docker CLI

You can build and run the container manually with standard Docker commands:

### Build Image
```bash
docker build -f docker/Dockerfile -t student-detection-api:latest .
```

### Run with GPU Support and Volume Mounts
```bash
docker run -d \
  --name student-detection-api \
  --gpus all \
  -p 8080:8080 \
  -v "$(pwd)/data/models:/app/data/models:ro" \
  -v student-detection-hf-cache:/root/.cache/huggingface \
  --restart unless-stopped \
  student-detection-api:latest
```

---

## 4. Running Without a GPU (CPU Mode)

If deploying to a CPU-only environment:

1. Comment out or remove the `deploy.resources.reservations` block in `docker/docker-compose.yml`.
2. Start the container:
```bash
docker compose -f docker/docker-compose.yml up -d
```

Or with Docker CLI:
```bash
docker run -d \
  --name student-detection-api \
  -p 8080:8080 \
  -v "$(pwd)/data/models:/app/data/models:ro" \
  student-detection-api:latest
```

---

## 5. Verification and Testing

### 1. Check API Documentation
Open your browser at [http://localhost:8080/docs](http://localhost:8080/docs) to access Swagger UI.

### 2. Send a Prediction Request
Test the `/api/predict` endpoint using `curl` with a test image:

```bash
curl -X POST "http://localhost:8080/api/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@data/samples/classroom.png;type=image/png"
```

Expected JSON response:
```json
{
  "id": "c1f7b7ad-4520-4e56-b08e-1f7c1184df9a",
  "people_count": 44,
  "kept_count": 44,
  "suppressed_count": 0,
  "detections": [...]
}
```

---

## 6. Configuration & Environment Variables

You can customize runtime behavior by passing environment variables:

| Variable | Default | Description |
|:---|:---|:---|
| `PORT` | `8080` | Host port mapped to container port 8080 |
| `DET_MODEL_PATH` | `data/models/yolo26m_finetuned_heads.pt` | Path to detection weights inside container |
| `DEPTH_CHECKPOINT` | `depth-anything/Depth-Anything-V2-Small-hf` | Hugging Face model ID or local directory |
| `PYTHONUNBUFFERED` | `1` | Forces unbuffered stdout and stderr logging |
