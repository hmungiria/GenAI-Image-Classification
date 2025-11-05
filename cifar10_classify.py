#!/usr/bin/env python3
"""
CIFAR-10 classification using ai.sooners.us (OpenAI-compatible Chat Completions).

Tasks:
  • Sample 100 images (10/class) from CIFAR-10
  • Send each as base64 to /api/chat/completions with gemma3:4b
  • Collect predictions, compute accuracy, and plot confusion matrix

Requires:
  pip install requests python-dotenv torch torchvision pillow scikit-learn matplotlib
"""

import os
import io
import time
import base64
import random
import json
import csv
import argparse
from typing import List, Dict, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision.datasets import CIFAR10

import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load secrets ──────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.expanduser("~"), ".soonerai.env"))
API_KEY = os.getenv("SOONERAI_API_KEY")
BASE_URL = os.getenv("SOONERAI_BASE_URL", "https://ai.sooners.us").rstrip("/")
MODEL = os.getenv("SOONERAI_MODEL", "gemma3:4b")

if not API_KEY:
    raise RuntimeError("Missing SOONERAI_API_KEY in ~/.soonerai.env")

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_SEED = 1337
DEFAULT_SAMPLES_PER_CLASS = 10
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

SYSTEM_PROMPT = """
Classify CIFAR-10 images into exactly one of:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
Reply with just the label, nothing else.
""".strip()

# Constrain the model’s output to *one* of the valid labels.
USER_INSTRUCTION = f"""
Classify this CIFAR-10 image. Respond with exactly one label from this list:
{', '.join(CLASSES)}
Your reply must be just the label, nothing else.
""".strip()

# ── Helpers ──────────────────────────────────────────────────────────────────
def pil_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    """Encode a PIL image to base64 JPEG data URL."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def post_chat_completion_image(
    image_data_url: str,
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    timeout: int = 60,
    max_retries: int = 3,
    retry_backoff: float = 1.5,
) -> str:
    """
    Send an image + instruction to /api/chat/completions and return the text reply.

    Uses OpenAI-style content parts with an image_url Data URL, which most
    OpenAI-compatible endpoints support for VLM inputs.
    """
    url = f"{base_url}/api/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }

    last_resp = None
    for attempt in range(max_retries):
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        last_resp = resp

        # Retry on transient server errors or rate limit
        if resp.status_code in (429, 500, 502, 503, 504):
            delay = (retry_backoff ** attempt)
            time.sleep(delay)
            continue

        if resp.status_code != 200:
            raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    # If exit due to repeated transient failures:
    raise RuntimeError(f"API transient failure after {max_retries} attempts. Last: {last_resp.status_code if last_resp else 'no response'}")

def normalize_label(text: str) -> str:
    """Map model reply to a valid CIFAR-10 class if possible (simple heuristic)."""
    t = text.lower().strip()
    # remove quotes and punctuation noise
    t = t.replace('"', '').replace("'", "").replace(".", "").replace(",", "")
    # exact match first
    if t in CLASSES:
        return t
    # loose matching: pick first class name contained in output
    for c in CLASSES:
        if c in t:
            return c
    # fallback: unknown (will count as incorrect)
    return "__unknown__"

# ── Data: stratified sample of N images (K/class) ────────────────────────────
def stratified_sample_cifar10(
    seed: int = DEFAULT_SEED,
    samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS,
    root: str = "./data"
) -> List[Tuple[Image.Image, int]]:
    """
    Download CIFAR-10 (train split) and return a list of (PIL_image, target) pairs:
    exactly samples_per_class per class.
    """
    ds = CIFAR10(root=root, train=True, download=True)
    # Build indices per class
    per_class: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(ds):
        per_class[label].append(idx)

    # Sample with fixed seed
    random.seed(seed)
    selected = []
    for label in range(10):
        chosen = random.sample(per_class[label], samples_per_class)
        for idx in chosen:
            img, tgt = ds[idx]
            selected.append((img, tgt))
    return selected

def plot_and_save_confusion_matrix(cm, classes: List[str], title: str, out_path: str):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            plt.text(c, r, str(cm[r, c]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 classification with ai.sooners.us VLM (gemma3:4b).")
    parser.add_argument("--system-prompt-file", type=str, default=None,
                        help="Path to a custom system prompt file; overrides the built-in SYSTEM_PROMPT.")
    parser.add_argument("--prompt-name", type=str, default=None,
                        help="A tag to identify outputs (e.g., baseline, defs).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for sampling.")
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS,
                        help="Images per class to sample (default 10).")
    parser.add_argument("--data-root", type=str, default="./data", help="Where to download CIFAR-10.")
    parser.add_argument("--out-dir", type=str, default="./outputs", help="Where to write outputs.")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality for base64 encoding.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the model.")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout (seconds).")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries on transient errors.")
    parser.add_argument("--retry-backoff", type=float, default=1.5, help="Exponential backoff base.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between requests to avoid rate limits.")
    parser.add_argument("--show", action="store_true",
                        help="Also display the confusion matrix window if a GUI backend is available.")
    args = parser.parse_args()

    # Resolve system prompt + output tag
    system_prompt = SYSTEM_PROMPT
    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    name_tag = args.prompt_name or ("file" if args.system_prompt_file else "default")

    os.makedirs(args.out_dir, exist_ok=True)

    total_images = args.samples_per_class * 10
    print(f"Preparing CIFAR-10 sample ({args.samples_per_class} images/class; total={total_images})...")
    samples = stratified_sample_cifar10(args.seed, args.samples_per_class, root=args.data_root)

    y_true: List[int] = []
    y_pred: List[int] = []
    rows: List[Dict] = []
    misclassified: List[Dict] = []

    print("Classifying...")
    for i, (img, tgt) in enumerate(samples, start=1):
        data_url = pil_to_base64_jpeg(img, quality=args.jpeg_quality)
        try:
            reply = post_chat_completion_image(
                image_data_url=data_url,
                system_prompt=system_prompt,
                model=MODEL,
                base_url=BASE_URL,
                api_key=API_KEY,
                temperature=args.temperature, 
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
            )
            pred_label = normalize_label(reply)
            pred_idx = CLASSES.index(pred_label) if pred_label in CLASSES else -1
        except Exception as e:
            reply = f"__error__: {e}"
            pred_label = "__error__"
            pred_idx = -1

        y_true.append(tgt)
        y_pred.append(pred_idx)

        true_label = CLASSES[tgt]
        print(f"[{i:03d}/{total_images}] true={true_label:>10s} | pred={pred_label:>10s} | raw='{reply}'")

        row = {
            "i": i,
            "true_idx": tgt,
            "true_label": true_label,
            "pred_idx": pred_idx,
            "pred_label": pred_label,
            "raw_reply": reply,
        }
        rows.append(row)

        if pred_idx != tgt:
            misclassified.append(row)

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Scoring
    y_pred_fixed = [p if 0 <= p < 10 else 9 for p in y_pred]

    acc = accuracy_score(y_true, y_pred_fixed)
    print(f"\nAccuracy over {total_images} images: {acc*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_fixed, labels=list(range(10)))
    cm_path = os.path.join(args.out_dir, f"confusion_matrix_{name_tag}.png")
    plot_and_save_confusion_matrix(
        cm, CLASSES,
        title=f"CIFAR-10 Confusion Matrix ({MODEL}, prompt={name_tag})",
        out_path=cm_path
    )
    print(f"Saved confusion matrix to: {os.path.abspath(cm_path)}")

    if args.show:
        try:
            import matplotlib.image as mpimg
            plt.figure(figsize=(8, 7))
            plt.imshow(mpimg.imread(cm_path))
            plt.axis('off')
            plt.show()
        except Exception:
            pass

    # Save detailed predictions to CSV
    csv_path = os.path.join(args.out_dir, f"predictions_{name_tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["i","true_idx","true_label","pred_idx","pred_label","raw_reply"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved predictions CSV to: {os.path.abspath(csv_path)}")

    # Save misclassifications as JSONL
    bad_path = os.path.join(args.out_dir, f"misclassifications_{name_tag}.jsonl")
    with open(bad_path, "w", encoding="utf-8") as f:
        for row in misclassified:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(misclassified)} misclassifications to: {os.path.abspath(bad_path)}")

if __name__ == "__main__":
    main()
