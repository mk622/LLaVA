FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 基本ツール + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

# pip を最新化
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# PyTorch(CUDA 12.1) 用の index を追加で参照
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

# 依存
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# エントリポイント
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# デフォルトは vLLM OpenAI互換API サーバ
ENTRYPOINT ["/app/entrypoint.sh"]
