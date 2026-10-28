#!/usr/bin/env python3
"""
Download nikolasgegenava/sneakers-classification from Kaggle and write a
stratified subsample (~500–1000 images) with one folder per sneaker model.

Requires a Kaggle API token at ~/.kaggle/kaggle.json
  1. https://www.kaggle.com/settings → Create New Token
  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
  3. chmod 600 ~/.kaggle/kaggle.json

Usage:
  python scripts/download_subsample.py
  python scripts/download_subsample.py --target 800 --max-per-class 20
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "sneakers-dataset-sample"
DATASET_SLUG = "nikolasgegenava/sneakers-classification"
ZIP_NAME = "sneakers-classification.zip"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def model_id_to_display(model_id: str) -> str:
    """adidas_dunk_low → Adidas Dunk Low (light brand capitalization)."""
    parts = model_id.replace("-", "_").split("_")
    special = {
        "nike": "Nike",
        "adidas": "Adidas",
        "yeezy": "Yeezy",
        "converse": "Converse",
        "vans": "Vans",
        "puma": "Puma",
        "reebok": "Reebok",
        "asics": "Asics",
        "salomon": "Salomon",
        "new": "New",
        "balance": "Balance",
        "air": "Air",
        "max": "Max",
        "jordan": "Jordan",
        "force": "Force",
        "dunk": "Dunk",
        "blazer": "Blazer",
        "cortez": "Cortez",
        "vapormax": "Vapormax",
        "ultraboost": "Ultraboost",
        "nmd": "NMD",
        "samba": "Samba",
        "gazelle": "Gazelle",
        "superstar": "Superstar",
        "stan": "Stan",
        "smith": "Smith",
        "forum": "Forum",
        "chuck": "Chuck",
        "taylor": "Taylor",
        "all": "All",
        "star": "Star",
        "sk8": "Sk8",
        "hi": "Hi",
        "gel": "Gel",
        "lyte": "Lyte",
        "iii": "III",
        "xt": "XT",
        "tn": "Tn",
        "r1": "R1",
        "v2": "V2",
        "77": "'77",
    }
    out = []
    for p in parts:
        key = p.lower()
        if key in special:
            out.append(special[key])
        elif p.isdigit():
            out.append(p)
        else:
            out.append(p[:1].upper() + p[1:].lower() if p else p)
    return " ".join(out)


def ensure_kaggle_cli() -> None:
    try:
        subprocess.run(
            [sys.executable, "-m", "kaggle", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise SystemExit(
            "Kaggle CLI not available. Install with: pip install kaggle\n"
            f"Detail: {e}"
        )


def check_credentials() -> Path:
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.is_file():
        raise SystemExit(
            "Missing ~/.kaggle/kaggle.json\n"
            "Create a token at https://www.kaggle.com/settings → API → Create New Token,\n"
            "then: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
        )
    return cred


def download_dataset(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / ZIP_NAME
    extract_marker = RAW_DIR / "sneakers-dataset"
    if extract_marker.is_dir() and not force:
        print(f"Using existing extract: {extract_marker}")
        return extract_marker
    if zip_path.is_file() and not force:
        print(f"Using existing zip: {zip_path}")
    else:
        print(f"Downloading {DATASET_SLUG} …")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                DATASET_SLUG,
                "-p",
                str(RAW_DIR),
                "--force",
            ],
            check=True,
        )
        # Kaggle may name the zip after the dataset slug basename
        if not zip_path.is_file():
            zips = list(RAW_DIR.glob("*.zip"))
            if not zips:
                raise SystemExit(f"Download finished but no zip found in {RAW_DIR}")
            zip_path = zips[0]
    print(f"Extracting {zip_path} …")
    if extract_marker.exists():
        shutil.rmtree(extract_marker)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    # Locate the model-folder root
    candidates = [
        RAW_DIR / "sneakers-dataset",
        RAW_DIR / "sneakers_dataset",
        RAW_DIR,
    ]
    for cand in candidates:
        if cand.is_dir() and any(p.is_dir() and p.name not in {".", ".."} for p in cand.iterdir()):
            # Prefer a dir that contains multiple class folders with images
            class_dirs = [
                d
                for d in cand.iterdir()
                if d.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in d.rglob("*") if f.is_file())
            ]
            if len(class_dirs) >= 5:
                print(f"Dataset root: {cand} ({len(class_dirs)} model folders)")
                return cand
    # Fallback: search one level deeper
    for sub in RAW_DIR.rglob("*"):
        if not sub.is_dir():
            continue
        class_dirs = [
            d
            for d in sub.iterdir()
            if d.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in d.iterdir() if f.is_file())
        ]
        if len(class_dirs) >= 5:
            print(f"Dataset root: {sub} ({len(class_dirs)} model folders)")
            return sub
    raise SystemExit(f"Could not find model-level folders under {RAW_DIR}")


def collect_images(dataset_root: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        imgs = [
            f
            for f in sorted(child.rglob("*"))
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ]
        if imgs:
            by_class[child.name].extend(imgs)
    if not by_class:
        raise SystemExit(f"No class folders with images under {dataset_root}")
    return dict(by_class)


def stratified_subsample(
    by_class: dict[str, list[Path]],
    target: int,
    max_per_class: int,
    seed: int,
) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    classes = sorted(by_class.keys())
    n_classes = len(classes)
    # Aim for even share, capped
    per = min(max_per_class, max(1, target // max(n_classes, 1)))
    selected: dict[str, list[Path]] = {}
    total = 0
    for cls in classes:
        pool = list(by_class[cls])
        rng.shuffle(pool)
        take = min(per, len(pool))
        selected[cls] = pool[:take]
        total += take
    # If under target, round-robin add more (still capped by available / max_per_class)
    if total < target:
        while total < target:
            grew = False
            for cls in classes:
                if total >= target:
                    break
                have = len(selected[cls])
                avail = by_class[cls]
                if have >= max_per_class or have >= len(avail):
                    continue
                # pick next unused
                used = set(selected[cls])
                for p in avail:
                    if p not in used:
                        selected[cls].append(p)
                        total += 1
                        grew = True
                        break
            if not grew:
                break
    # If over target (shouldn't with per calc), trim randomly
    if total > target:
        flat = [(c, p) for c, ps in selected.items() for p in ps]
        rng.shuffle(flat)
        flat = flat[:target]
        trimmed: dict[str, list[Path]] = defaultdict(list)
        for c, p in flat:
            trimmed[c].append(p)
        selected = dict(trimmed)
        total = sum(len(v) for v in selected.values())
    print(
        f"Subsample: {total} images across {len(selected)} classes "
        f"(~{total // max(len(selected), 1)} per class, max_per_class={max_per_class})"
    )
    return selected


def write_sample(selected: dict[str, list[Path]], out_dir: Path) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    index = []
    for model_id, paths in sorted(selected.items()):
        dest_dir = out_dir / model_id
        dest_dir.mkdir(parents=True)
        display = model_id_to_display(model_id)
        for i, src in enumerate(paths):
            # Preserve extension; unique name within class
            dest_name = f"{model_id}_{i:04d}{src.suffix.lower()}"
            dest = dest_dir / dest_name
            shutil.copy2(src, dest)
            rel = f"{model_id}/{dest_name}"
            index.append(
                {
                    "path": rel,
                    "model_id": model_id,
                    "display_name": display,
                    "filename": dest_name,
                }
            )
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    meta = {
        "dataset": DATASET_SLUG,
        "dataset_url": f"https://www.kaggle.com/datasets/{DATASET_SLUG}",
        "license": "MIT",
        "n_images": len(index),
        "n_classes": len(selected),
        "classes": sorted(selected.keys()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(index)} images → {out_dir}")
    print(f"Index: {index_path}")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download + subsample Kaggle sneakers dataset")
    parser.add_argument("--target", type=int, default=800, help="Target total images (500–1000)")
    parser.add_argument("--max-per-class", type=int, default=20, help="Cap images per model folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=SAMPLE_DIR,
        help="Output sample directory (default: sneakers-dataset-sample/)",
    )
    args = parser.parse_args()
    if args.target < 100 or args.target > 5000:
        raise SystemExit("--target should be a sensible count (e.g. 500–1000)")

    ensure_kaggle_cli()
    check_credentials()
    root = download_dataset(force=args.force_download)
    by_class = collect_images(root)
    print(f"Found {len(by_class)} classes, {sum(len(v) for v in by_class.values())} images raw")
    selected = stratified_subsample(by_class, args.target, args.max_per_class, args.seed)
    write_sample(selected, args.out.resolve())
    print("Done. Next: python demo_ui.py")


if __name__ == "__main__":
    main()
