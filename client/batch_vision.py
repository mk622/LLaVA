#!/usr/bin/env python3
import os, sys, json, time, shutil, glob, re, argparse, base64, mimetypes
from typing import Any, Dict, Optional
import yaml, requests

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(*dirs: str):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def encode_image_as_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    import re, json
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=None, help="画像のルート（デフォルトは config.yaml の input_dir）")
    ap.add_argument("--output", default=None, help="JSONL の出力パス（指定時は1行/画像で追記）")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))

    base_url   = cfg["openai_base_url"].rstrip("/")
    api_key    = cfg.get("api_key", "EMPTY")
    model      = cfg["model"]

    input_dir  = args.input_dir or cfg["input_dir"]
    input_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", input_dir))
    json_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", cfg["json_out_dir"]))
    out_true   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", cfg["image_out_true_dir"]))
    out_false  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", cfg["image_out_false_dir"]))
    ensure_dirs(json_dir, out_true, out_false)

    instruction = cfg["instruction"]
    temperature = float(cfg.get("temperature", 0))
    max_tokens  = int(cfg.get("max_tokens", 256))
    retry       = int(cfg.get("retry", 2))

    images = sorted(glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True))
    if not images:
        print(f"No .jpg found in: {input_dir}")
        sys.exit(0)

    print(f"Found {len(images)} images. Start processing...")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = f"{base_url}/chat/completions"

    out_jsonl = open(args.output, "a", encoding="utf-8") if args.output else None

    for idx, host_img in enumerate(images, 1):
        data_url = encode_image_as_data_url(host_img)
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }]
        }

        data = None
        for attempt in range(retry + 1):
            try:
                r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt >= retry:
                    print(f"[{idx}/{len(images)}] ERROR request failed: {host_img} -> {e}")
                time.sleep(1.5 * (attempt + 1))

        if not data:
            shutil.copy2(host_img, os.path.join(out_false, os.path.basename(host_img)))
            continue

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            content = ""

        parsed: Optional[Dict[str, Any]] = None
        if content:
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = extract_json(content)
        if not parsed or not isinstance(parsed, dict):
            parsed = {"is_forbidden": False, "reason": "parse_error", "point": [0, 0]}

        is_forbidden = bool(parsed.get("is_forbidden", False))

        stem = os.path.splitext(os.path.basename(host_img))[0]
        json_path = os.path.join(json_dir, stem + ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)

        dst_dir = out_true if is_forbidden else out_false
        shutil.copy2(host_img, os.path.join(dst_dir, os.path.basename(host_img)))

        if out_jsonl:
            out_jsonl.write(json.dumps({
                "image": os.path.relpath(host_img, input_dir),
                "result": parsed
            }, ensure_ascii=False) + "\n")

        print(f"[{idx}/{len(images)}] {os.path.basename(host_img)} -> {parsed}")

    if out_jsonl:
        out_jsonl.close()
    print("Done.")

if __name__ == "__main__":
    main()
