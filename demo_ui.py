#!/usr/bin/env python3
"""Simple Gradio UI for sneakers CLIP ranking demo."""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import open_clip
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "images"
INDEX = json.loads((ROOT / "index.json").read_text(encoding="utf-8"))
DEVICE = "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(DEVICE).eval()

paths = [IMG_DIR / r["filename"] for r in INDEX]
tensors = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(DEVICE)
with torch.no_grad():
    img_feats = model.encode_image(tensors)
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
img_feats = img_feats.cpu()


def search(query: str, k: int = 5):
    query = (query or "").strip()
    if not query:
        return [], "Type a text query, e.g. red Nike sneakers"
    tokens = tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        t = model.encode_text(tokens)
        t = t / t.norm(dim=-1, keepdim=True)
    t = t.cpu()
    scores = (img_feats @ t.T).squeeze(1)
    top = torch.topk(scores, k=min(k, len(INDEX)))
    gallery = []
    lines = [f"Query: {query}", ""]
    for rank, (idx, score) in enumerate(zip(top.indices.tolist(), top.values.tolist()), 1):
        row = INDEX[idx]
        path = IMG_DIR / row["filename"]
        caption = f"{rank}. {row['brand']} / {row['colour']}  ({score:.3f})"
        gallery.append((str(path), caption))
        lines.append(
            f"{rank}. {score:.3f}  {row['filename']}  [{row['brand']} / {row['colour']}]"
        )
        lines.append(f"    {row['caption']}")
    return gallery, "\n".join(lines)


demo = gr.Blocks(title="COMP5523 Sneakers Text→Image Demo")
with demo:
    gr.Markdown(
        "# COMP5523 — Sneakers text-to-image sample\n"
        "20-photo gallery · CLIP ViT-B/32 · brand + colour"
    )
    with gr.Row():
        q = gr.Textbox(
            label="Text query",
            value="red Nike sneakers",
            placeholder="e.g. white Converse, black Adidas...",
        )
        btn = gr.Button("Search", variant="primary")
    out_gal = gr.Gallery(label="Top matches", columns=5, height=320, object_fit="contain")
    out_txt = gr.Textbox(label="Ranking", lines=14)
    examples = gr.Examples(
        examples=[
            ["red Nike sneakers"],
            ["white Converse sneakers"],
            ["black Adidas sneakers"],
            ["pink Puma sneakers"],
        ],
        inputs=q,
    )
    btn.click(search, inputs=q, outputs=[out_gal, out_txt])
    q.submit(search, inputs=q, outputs=[out_gal, out_txt])
    demo.load(lambda: search("red Nike sneakers"), outputs=[out_gal, out_txt])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)
