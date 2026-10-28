# COMP5523 — Sneakers text→image (CLIP + Gradio)

CLIP **ViT-B/32** text→image ranking demo for COMP5523. Gallery images come from the Kaggle dataset **[Popular Sneakers Classification](https://www.kaggle.com/datasets/nikolasgegenava/sneakers-classification)** (`nikolasgegenava/sneakers-classification`) — ~50 sneaker **model** folders (`nike_dunk_low`, `adidas_samba`, …) with 5k+ photos total.

This repo does **not** ship the images. A script downloads the dataset via the Kaggle API and writes a stratified subsample of **~500–1000** images (default **800**, capped per class) under `sneakers-dataset-sample/`.

**License / attribution:** dataset is **MIT** on Kaggle (see the [dataset card](https://www.kaggle.com/datasets/nikolasgegenava/sneakers-classification)). Cite Nikolas Gegenava / that Kaggle page if you redistribute results.

## Quick start

```bash
# 1) Python env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Kaggle API token (one-time)
#    https://www.kaggle.com/settings → API → Create New Token
mkdir -p ~/.kaggle
# move the downloaded kaggle.json into place, then:
chmod 600 ~/.kaggle/kaggle.json

# 3) Download + subsample (~800 images across model folders)
python scripts/download_subsample.py
# optional: python scripts/download_subsample.py --target 600 --max-per-class 15

# 4) Gradio UI (port 7860)
python demo_ui.py
```

CLI ranking (no UI):

```bash
python demo_clip_rank.py --query "red nike dunk low" --top 8
```

## Dataset layout

After `download_subsample.py`:

```
sneakers-dataset-sample/
  adidas_samba/
    adidas_samba_0000.jpg
    …
  nike_dunk_low/
    …
  index.json          # path, model_id, display_name per image
  meta.json           # counts + class list
  image_features.pt   # optional CLIP cache written by demo_ui.py
```

Raw Kaggle extract (ignored by git) lives under `data/raw/`.

## Repo contents

| Path | Purpose |
|------|---------|
| `demo_ui.py` | Gradio text→image UI (CLIP ViT-B/32, CPU) |
| `demo_clip_rank.py` | Headless CLIP ranker |
| `scripts/download_subsample.py` | Kaggle download + stratified subsample |
| `requirements.txt` | Python deps (incl. `kaggle`) |
| `.gitignore` | Ignores `data/`, sample images, venv, caches |

## Notes

- First CLIP run downloads OpenAI ViT-B/32 weights; needs network + disk/RAM.
- `demo_ui.py` caches image embeddings in `sneakers-dataset-sample/image_features.pt`.
- Subsample is stratified across model folders so rare classes still appear.
- Do not commit `sneakers-dataset-sample/` or `data/` — they are gitignored on purpose.
