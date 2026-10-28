#!/usr/bin/env python3
"""
Tiny CLIP ranking demo for the COMP5523 sneakers sample.

Embeds gallery images + a few text queries with open_clip (preferred) or
transformers CLIP, then prints cosine-similarity rankings.

Usage:
  python demo_clip_rank.py
  python demo_clip_rank.py --query "red Nike sneakers"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "images"
INDEX = ROOT / "index.json"


def load_index():
    rows = json.loads(INDEX.read_text(encoding="utf-8"))
    missing = [r["filename"] for r in rows if not (IMG_DIR / r["filename"]).exists()]
    if missing:
        raise SystemExit(f"Missing images: {missing}")
    return rows


def try_open_clip(device: str):
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    def embed_images(paths):
        tensors = []
        for p in paths:
            tensors.append(preprocess(Image.open(p).convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()

    def embed_texts(texts):
        tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()

    return embed_images, embed_texts, "open_clip ViT-B-32 (openai)"


def try_transformers(device: str):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    def embed_images(paths):
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()

    def embed_texts(texts):
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()

    return embed_images, embed_texts, f"transformers {model_id}"


def main():
    parser = argparse.ArgumentParser(description="CLIP rank sneakers gallery")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Text query (repeatable). Defaults to a small built-in set.",
    )
    parser.add_argument("--top", type=int, default=5, help="Top-K results per query")
    args = parser.parse_args()

    queries = args.queries or [
        "red Nike running sneaker on white background",
        "black Puma suede classic sneaker",
        "white Converse Chuck Taylor high-top",
    ]

    rows = load_index()
    paths = [IMG_DIR / r["filename"] for r in rows]

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    embed_images = embed_texts = backend = None
    errors = []
    for loader, name in ((try_open_clip, "open_clip"), (try_transformers, "transformers")):
        try:
            embed_images, embed_texts, backend = loader(device)
            print(f"Using backend: {backend} on {device}")
            break
        except Exception as e:
            errors.append(f"{name}: {e}")

    if embed_images is None:
        print("CLIP demo skipped — could not load open_clip or transformers CLIP.")
        for e in errors:
            print(f"  - {e}")
        print("Install tip: pip install open-clip-torch  OR  pip install transformers torch pillow")
        sys.exit(0)

    import torch

    print(f"Embedding {len(paths)} gallery images...")
    img_feats = embed_images(paths)
    txt_feats = embed_texts(queries)

    sims = (txt_feats @ img_feats.T).numpy()  # [Q, N]
    for qi, q in enumerate(queries):
        print(f"\n=== Query: {q!r} ===")
        order = sims[qi].argsort()[::-1][: args.top]
        for rank, idx in enumerate(order, 1):
            r = rows[idx]
            print(
                f"  {rank}. {sims[qi, idx]:.3f}  {r['filename']:28s}  "
                f"[{r['brand']} / {r['colour']}]  {r['caption']}"
            )


if __name__ == "__main__":
    main()
