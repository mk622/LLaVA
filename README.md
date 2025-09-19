# LLaVA

## Overview
This repository wraps the LLaVA 1.5 7B vision-language model with a vLLM OpenAI-compatible API server and provides tools for batch image inspection.

## Requirements
- Docker 24+ with the Compose plugin
- NVIDIA GPU with 24 GB memory (tested with CUDA 12.1 runtime)
- NVIDIA Container Toolkit configured for Docker
- (Optional) Hugging Face token with access to llava-hf models

## Directory Layout
- `docker-compose.yml` – main stack exposing the API on `http://localhost:9000/v1`
- `vllm/` – Docker build context for the API server
- `client/` – lightweight batch inspector that calls the API
- `data/` – default input/output folders used by the client
- `hf_cache/` – shared Hugging Face cache volume for downloaded models
- `bu/` – legacy compose stack that runs the inspector inside containers

## Quick Start
1. Clone this repository and switch into the project root.
2. (Optional) export `HF_TOKEN` so the container can download gated models: `export HF_TOKEN=...`
3. Build the vLLM image: `docker compose build`
4. Launch the stack: `docker compose up -d`
5. Tail logs until the health check passes: `docker compose logs -f vllm`
6. Verify the API: `curl http://localhost:9000/v1/models`
7. Stop and remove containers when finished: `docker compose down`

### Model Cache
- The first start downloads the model into `hf_cache/`. Preserve this directory to avoid re-downloading.
- `data/` is mounted read-only into the server. Place reference assets there if you want to test `file://` image URLs.

## Batch Inspector (client/)
1. Create a local Python environment (`python -m venv .venv && source .venv/bin/activate`).
2. Install dependencies: `pip install -r client/requirements.txt`
3. Adjust `client/config.yaml` if you need to change the API URL, model name, or output locations.
4. Put `.jpg` files under `data/in/`; the script writes JSON results to `data/json/` and copies annotated hits into `data/out_true/` or `data/out_false/`.
5. Run `python client/batch_vision.py`. The script retries transient API errors, enforces the allowed label set, and draws bounding boxes on positive detections.

## Legacy docker-compose stack (bu/)
- `docker compose -f bu/docker-compose.yml up --build` spins up both the vLLM service and the in-container inspector defined in `bu/app/`.
- Place source images in `bu/data/images/`; results are collected under `bu/app/outputs/`.
- This stack is useful when you want a single command workflow but is not required for local development.

## Troubleshooting
- Ensure the host driver and CUDA runtime match (the base image uses CUDA 12.1).
- If downloads from Hugging Face fail, confirm that `HF_TOKEN` is available inside the container (`docker compose exec vllm env | grep HF_TOKEN`).
- Large batches may require tuning `--gpu-memory-utilization` or `--max-num-seqs` in `docker-compose.yml`.
