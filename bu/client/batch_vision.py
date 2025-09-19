#!/usr/bin/env python3
import os
import sys
import time
import json
import base64
from pathlib import Path
import requests

# ===== 設定 =====
API_BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:9000")
MODEL_ID = os.environ.get("VLLM_MODEL", "llava-hf/llava-1.5-7b-hf")
IN_DIR = Path("data/in")     # ホスト側の入力ディレクトリ
OUT_DIR = Path("data/out")   # ホスト側の出力ディレクトリ
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ホストの data/ は コンテナでは /data で見える（read-only マウント）
CONTAINER_DATA_PREFIX = Path("/data")  # コンテナ内から見えるルート

# 拡張子
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}

# プロンプト（必要に応じて編集）
DEFAULT_PROMPT = "この画像の要点を日本語で簡潔に説明してください。"

# タイムアウト/リトライ
HTTP_TIMEOUT = 120
RETRY = 3
BACKOFF = 2.0


def to_container_file_url(host_path: Path) -> str:
    """
    ホストの data/in/xxx.jpg -> コンテナでは /data/in/xxx.jpg
    -> image_url として 'file:///data/in/xxx.jpg'
    """
    rel = host_path.resolve().relative_to(Path("data").resolve())
    container_path = CONTAINER_DATA_PREFIX / rel
    return f"file://{container_path.as_posix()}"


def iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in IMG_EXTS:
            yield p


def request_one(image_path: Path) -> dict:
    url = f"{API_BASE}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    image_url = to_container_file_url(image_path)
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": DEFAULT_PROMPT},
                ],
            }
        ],
        "temperature": 0.2,
        "max_tokens": 256
    }

    last_err = None
    for i in range(1, RETRY + 1):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            else:
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
        # backoff
        time.sleep(BACKOFF * i)

    raise last_err


def main():
    imgs = list(iter_images(IN_DIR))
    if not imgs:
        print(f"No images found in: {IN_DIR}")
        return

    print(f"Found {len(imgs)} images. Start processing...")

    for idx, p in enumerate(imgs, 1):
        out_path = OUT_DIR / (p.stem + ".json")
        if out_path.exists():
            print(f"[{idx}/{len(imgs)}] skip (exists): {p}")
            continue

        try:
            res = request_one(p)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"[{idx}/{len(imgs)}] OK: {p} -> {out_path}")
        except Exception as e:
            print(f"[{idx}/{len(imgs)}] ERROR request failed: {p} -> {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
