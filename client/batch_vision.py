#!/usr/bin/env python3
import os, sys, json, time, shutil, glob, re, argparse, base64, mimetypes
from io import BytesIO
from typing import Any, Dict, Optional, Tuple, List
import yaml, requests

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(*dirs: str):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

THUMB_MAX_DIM = 512


def encode_image_as_data_url(path: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    mime, _ = mimetypes.guess_type(path)
    size: Optional[Tuple[int, int]] = None
    image_bytes: Optional[bytes] = None
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as img:  # type: ignore
            size = img.size
            working = img
            if max(img.size) > THUMB_MAX_DIM:
                working = img.copy()
                working.thumbnail((THUMB_MAX_DIM, THUMB_MAX_DIM), Image.LANCZOS)
            if working.mode not in ("RGB", "L"):
                working = working.convert("RGB")
            buffer = BytesIO()
            working.save(buffer, format="JPEG", quality=85)
            image_bytes = buffer.getvalue()
            mime = "image/jpeg"
    except Exception:
        image_bytes = None

    if image_bytes is None:
        with open(path, "rb") as f:
            image_bytes = f.read()
        if not mime:
            mime = "image/jpeg"
        if size is None:
            try:
                from PIL import Image  # type: ignore
                with Image.open(path) as img:  # type: ignore
                    size = img.size
            except Exception:
                size = None

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}", size


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _clamp(val: int, low: int, high: int) -> int:
    return max(low, min(high, val))


def normalize_point(point: Any, image_size: Optional[Tuple[int, int]]) -> List[int]:
    if not isinstance(point, (list, tuple)) or len(point) < 2 or not image_size:
        return [0, 0]
    x_raw = _to_float(point[0])
    y_raw = _to_float(point[1])
    if x_raw is None or y_raw is None:
        return [0, 0]
    w, h = image_size
    if w <= 0 or h <= 0:
        return [0, 0]
    if 0.0 <= x_raw <= 1.0 and 0.0 <= y_raw <= 1.0:
        x = int(round(x_raw * (w - 1)))
        y = int(round(y_raw * (h - 1)))
    else:
        x = int(round(x_raw))
        y = int(round(y_raw))
    return [_clamp(x, 0, w - 1), _clamp(y, 0, h - 1)]


def normalize_bbox(bbox: Any, image_size: Optional[Tuple[int, int]]) -> List[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4 or not image_size:
        return [0, 0, 0, 0]
    vals = [_to_float(bbox[i]) for i in range(4)]
    if any(v is None for v in vals):
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = vals
    w, h = image_size
    if w <= 0 or h <= 0:
        return [0, 0, 0, 0]
    if all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
        x1 *= (w - 1)
        y1 *= (h - 1)
        x2 *= (w - 1)
        y2 *= (h - 1)
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 == x1 or y2 == y1:
        return [0, 0, 0, 0]
    x1 = _clamp(x1, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)
    max_x = max(x1 + 1, w - 1)
    max_y = max(y1 + 1, h - 1)
    x2 = _clamp(x2, x1 + 1, max_x)
    y2 = _clamp(y2, y1 + 1, max_y)
    if x2 == x1 or y2 == y1:
        return [0, 0, 0, 0]
    return [x1, y1, x2, y2]


def ensure_bbox(bbox: List[int], point: List[int], image_size: Optional[Tuple[int, int]]) -> List[int]:
    if bbox and any(bbox):
        return bbox
    if image_size and point and any(point):
        w, h = image_size
        x, y = point
        half = max(20, int(round(min(w, h) * 0.05)))
        x1 = _clamp(x - half, 0, w - 1)
        y1 = _clamp(y - half, 0, h - 1)
        x2 = _clamp(x + half, x1 + 1, w)
        y2 = _clamp(y + half, y1 + 1, h)
        return [x1, y1, x2, y2]
    return [0, 0, 0, 0]


def annotate_detection(src_path: str, dst_path: str, bbox: List[int], label: str):
    if not bbox or len(bbox) < 4 or bbox[0] == bbox[2] or bbox[1] == bbox[3]:
        shutil.copy2(src_path, dst_path)
        return
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        shutil.copy2(src_path, dst_path)
        return

    with Image.open(src_path) as img:  # type: ignore
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=max(2, int(round(min(img.size) * 0.004))))
        text = label.strip() if label else "target"
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(12, int(round(min(img.size) * 0.03))))
        except Exception:
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        padding = 4
        box_x1 = x1
        box_y1 = max(0, y1 - text_h - padding * 2)
        box_x2 = min(img.width, x1 + text_w + padding * 2)
        box_y2 = box_y1 + text_h + padding * 2
        draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill="red")
        draw.text((box_x1 + padding, box_y1 + padding), text, fill="white", font=font)
        img.save(dst_path)


def clean_response_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = cleaned.replace("\\_", "_")
    return cleaned


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    import re, json
    cleaned = clean_response_text(text)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
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

    system_prompt = cfg.get("system_prompt")
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
        data_url, image_size = encode_image_as_data_url(host_img)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        few_shot_examples = cfg.get("few_shot_examples") or []
        for example in few_shot_examples:
            user_text = example.get("user")
            assistant_text = example.get("assistant")
            if user_text:
                messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

        user_content = [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": instruction},
        ]
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages
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
            cleaned = clean_response_text(content)
            try:
                parsed = json.loads(cleaned)
            except Exception:
                parsed = extract_json(cleaned)
        if not parsed or not isinstance(parsed, dict):
            parsed = {"is_forbidden": False, "reason": "parse_error", "point": [0, 0], "bbox": [0, 0, 0, 0]}

        is_forbidden = bool(parsed.get("is_forbidden", False))

        if not isinstance(parsed.get("reason"), str):
            parsed["reason"] = str(parsed.get("reason", ""))

        parsed_point = normalize_point(parsed.get("point"), image_size)
        parsed_bbox = normalize_bbox(parsed.get("bbox"), image_size)
        if is_forbidden:
            parsed_bbox = ensure_bbox(parsed_bbox, parsed_point, image_size)
        else:
            parsed_point = [0, 0]
            parsed_bbox = [0, 0, 0, 0]

        parsed["point"] = parsed_point
        parsed["bbox"] = parsed_bbox

        stem = os.path.splitext(os.path.basename(host_img))[0]
        json_path = os.path.join(json_dir, stem + ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)

        dst_dir = out_true if is_forbidden else out_false
        dst_path = os.path.join(dst_dir, os.path.basename(host_img))
        if is_forbidden:
            try:
                annotate_detection(host_img, dst_path, parsed_bbox, parsed.get("reason", "target"))
            except Exception:
                shutil.copy2(host_img, dst_path)
        else:
            shutil.copy2(host_img, dst_path)

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
