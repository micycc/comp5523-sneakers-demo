# COMP5523 — Sneakers text→image sample

Small Gradio demo: type a text query (e.g. `red Nike sneakers`) and rank a 20-image sneaker gallery with **CLIP ViT-B/32** (OpenAI weights via `open_clip`). Index includes brand + colour.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python demo_ui.py
```

Opens a Gradio UI on port `7860` (and a share link when `share=True`).

CLI ranking without the UI:

```bash
python demo_clip_rank.py --query "red Nike sneakers"
```

## Layout

| Path | Purpose |
|------|---------|
| `demo_ui.py` | Gradio text→image ranking UI |
| `demo_clip_rank.py` | Headless CLIP ranking script |
| `images/` | 20 sneaker JPGs |
| `index.json` / `index.csv` | filename, brand, colour, caption, license |
| `LICENSE_NOTES.md` | Image source / license notes |

## Notes

- Images are Wikimedia Commons (see `LICENSE_NOTES.md` / index attribution).
- First run downloads CLIP weights; needs a working network and enough disk/RAM.
- CPU-only by default in `demo_ui.py`.
