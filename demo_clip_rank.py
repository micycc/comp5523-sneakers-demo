#!/usr/bin/env python3
"""
CLI CLIP ranker for the Kaggle sneakers subsample.

Usage:
  python demo_clip_rank.py
  python demo_clip_rank.py --query "red nike dunk low" --top 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sneakers-dataset-sample"
INDEX_PATH = SAMPLE_DIR / "index.json"
BATCH = 32


def load_index():
    if not INDEX_PATH.is_file():
        raise SystemExit(
            "Missing sneakers-dataset-sample/index.json — run scripts/download_subsample.py first"
        )
    rows = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    missing = [r["path"] for r in rows if not (SAMPLE_DIR / r["path"]).is_file()]
    if missing:
        raise SystemExit(f"Missing images ({len(missing)}). Re-run download_subsample.py")
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
        feats = []
        with torch.no_grad():
            for i in range(0, len(paths), BATCH):
                batch = paths[i : i + BATCH]
                tensors = torch.stack(
                    [preprocess(Image.open(p).convert("RGB")) for p in batch]
                ).to(device)
                f = model.encode_image(tensors)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu())
        return torch.cat(feats, dim=0)

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
        feats = []
        with torch.no_grad():
            for i in range(0, len(paths), BATCH):
                images = [Image.open(p).convert("RGB") for p in paths[i : i + BATCH]]
                inputs = processor(images=images, return_tensors="pt").to(device)
                f = model.get_image_features(**inputs)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu())
        return torch.cat(feats, dim=0)

    def embed_texts(texts):
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()

    return embed_images, embed_texts, f"transformers {model_id}"


def main():
    parser = argparse.ArgumentParser(description="CLIP rank sneakers subsample")
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    queries = args.queries or [
        "red nike dunk low",
        "white adidas samba",
        "black air jordan 1 high",
    ]

    rows = load_index()
    paths = [SAMPLE_DIR / r["path"] for r in rows]

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
        print("Install tip: pip install -r requirements.txt")
        sys.exit(0)

    import torch

    print(f"Embedding {len(paths)} gallery images…")
    img_feats = embed_images(paths)
    txt_feats = embed_texts(queries)
    sims = (txt_feats @ img_feats.T).numpy()
    for qi, q in enumerate(queries):
        print(f"\n=== Query: {q!r} ===")
        order = sims[qi].argsort()[::-1][: args.top]
        for rank, idx in enumerate(order, 1):
            r = rows[idx]
            print(
                f"  {rank}. {sims[qi, idx]:.3f}  [{r['model_id']:28s}]  "
                f"{r['display_name']}  {r['path']}"
            )


if __name__ == "__main__":
    main()
