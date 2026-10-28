#!/usr/bin/env python3
"""Gradio CLIP text→image ranking over the Kaggle sneakers subsample."""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import open_clip
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sneakers-dataset-sample"
INDEX_PATH = SAMPLE_DIR / "index.json"
DEVICE = "cpu"
BATCH = 32

if not INDEX_PATH.is_file():
    raise SystemExit(
        "Missing sneakers-dataset-sample/index.json\n"
        "Run: python scripts/download_subsample.py\n"
        "(requires ~/.kaggle/kaggle.json — see README)"
    )

INDEX = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
PATHS = [SAMPLE_DIR / row["path"] for row in INDEX]
missing = [str(p) for p in PATHS if not p.is_file()]
if missing:
    raise SystemExit(f"Missing {len(missing)} images (first: {missing[0]}). Re-run download_subsample.py")

print(f"Loading CLIP ViT-B/32 on {DEVICE} …")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(DEVICE).eval()

cache_path = SAMPLE_DIR / "image_features.pt"
if cache_path.is_file() and cache_path.stat().st_mtime >= INDEX_PATH.stat().st_mtime:
    print(f"Loading cached image features: {cache_path}")
    blob = torch.load(cache_path, map_location="cpu")
    if blob.get("n") == len(INDEX):
        img_feats = blob["feats"]
    else:
        img_feats = None
else:
    img_feats = None

if img_feats is None:
    print(f"Encoding {len(PATHS)} images (batched) …")
    feats = []
    with torch.no_grad():
        for i in range(0, len(PATHS), BATCH):
            batch_paths = PATHS[i : i + BATCH]
            tensors = torch.stack(
                [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            ).to(DEVICE)
            f = model.encode_image(tensors)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
            if (i // BATCH) % 5 == 0:
                print(f"  {min(i + BATCH, len(PATHS))}/{len(PATHS)}")
    img_feats = torch.cat(feats, dim=0)
    torch.save({"n": len(INDEX), "feats": img_feats}, cache_path)
    print(f"Cached features → {cache_path}")


def search(query: str, k: int = 8):
    query = (query or "").strip()
    if not query:
        return [], "Type a text query, e.g. red nike dunk low"
    k = int(k) if k else 8
    tokens = tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        t = model.encode_text(tokens)
        t = t / t.norm(dim=-1, keepdim=True)
    t = t.cpu()
    scores = (img_feats @ t.T).squeeze(1)
    top = torch.topk(scores, k=min(k, len(INDEX)))
    gallery = []
    lines = [f"Query: {query}", f"Gallery size: {len(INDEX)}", ""]
    for rank, (idx, score) in enumerate(zip(top.indices.tolist(), top.values.tolist()), 1):
        row = INDEX[idx]
        path = SAMPLE_DIR / row["path"]
        label = f"{rank}. {row['display_name']}  ({score:.3f})"
        gallery.append((str(path), label))
        lines.append(
            f"{rank}. {score:.3f}  [{row['model_id']}]  {row['display_name']}  {row['path']}"
        )
    return gallery, "\n".join(lines)


demo = gr.Blocks(title="COMP5523 Sneakers CLIP Demo")
with demo:
    gr.Markdown(
        "# COMP5523 — Sneakers text→image (CLIP)\n"
        f"Kaggle [Popular Sneakers Classification](https://www.kaggle.com/datasets/nikolasgegenava/sneakers-classification) "
        f"subsample · **{len(INDEX)}** images · CLIP ViT-B/32 · model-folder labels"
    )
    with gr.Row():
        q = gr.Textbox(
            label="Text query",
            value="red nike dunk low",
            placeholder="e.g. white adidas samba, black air jordan 1…",
            scale=4,
        )
        k = gr.Slider(1, 20, value=8, step=1, label="Top-K", scale=1)
        btn = gr.Button("Search", variant="primary", scale=1)
    out_gal = gr.Gallery(label="Top matches", columns=4, height=420, object_fit="contain")
    out_txt = gr.Textbox(label="Ranking", lines=16)
    gr.Examples(
        examples=[
            ["red nike dunk low"],
            ["white adidas samba"],
            ["black air jordan 1 high"],
            ["grey new balance 550"],
            ["beige yeezy boost 350"],
            ["checkerboard vans slip on"],
        ],
        inputs=q,
    )
    btn.click(search, inputs=[q, k], outputs=[out_gal, out_txt])
    q.submit(search, inputs=[q, k], outputs=[out_gal, out_txt])
    demo.load(lambda: search("red nike dunk low", 8), outputs=[out_gal, out_txt])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
