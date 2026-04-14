"""
bbbc020_prepare_data.py
=======================
Run this ONCE before using the Cellpose notebook.

What it does:
  1. Reads every per-cell outline TIF from BBC020_v1_outlines_cells/
     (each file = one binary mask for one cell)
  2. Merges them into a single integer instance mask per image
     (background=0, cell1=1, cell2=2, …)
  3. Copies the matching input image (c1 = DAPI channel, grayscale)
     into a flat input/ folder

Output layout (created automatically):
  project_cellpose-main/
  └── BBBC020_prepared/
      ├── images/          <- one grayscale TIF per image
      └── ground_truth/    <- one integer instance-mask TIF per image

Usage:
  python bbbc020_prepare_data.py
"""

import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import tifffile

# ── Paths — edit if needed ────────────────────────────────────────────────────
BASE       = Path("/Users/sorro/workspace/02604/project/project_cellpose-main")
IMG_ROOT   = BASE / "BBBC020_v1_images"        # contains jw-xxxx/ subfolders
GT_ROOT    = BASE / "BBC020_v1_outlines_cells"  # flat folder of per-cell TIFs

OUT_ROOT   = BASE / "BBBC020_prepared"
OUT_IMAGES = OUT_ROOT / "images"
OUT_GT     = OUT_ROOT / "ground_truth"

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_GT.mkdir(parents=True, exist_ok=True)

# ── Step 1: group per-cell outline files by image stem ───────────────────────
# Filename pattern: "jw-15min 1_c1_0.TIF"  →  stem="jw-15min 1_c1", idx=0
pattern = re.compile(r"^(.+)_(\d+)\.TIF$", re.IGNORECASE)

grouped = defaultdict(list)
for p in sorted(GT_ROOT.glob("*.TIF")):
    m = pattern.match(p.name)
    if m:
        stem, idx = m.group(1), int(m.group(2))
        grouped[stem].append((idx, p))

print(f"Found {len(grouped)} image stems in GT folder.")

# ── Step 2: for each stem, merge outlines → instance mask ────────────────────
for stem, cell_files in sorted(grouped.items()):
    # Sort by cell index so label assignment is deterministic
    cell_files.sort(key=lambda x: x[0])

    # Read first file to get image shape
    first = tifffile.imread(str(cell_files[0][1]))
    if first.ndim == 3:
        first = first[..., 0]          # RGB-stored binary → take one channel
    H, W = first.shape

    instance_mask = np.zeros((H, W), dtype=np.int32)
    n_cells = 0

    for idx, fpath in cell_files:
        outline = tifffile.imread(str(fpath))
        if outline.ndim == 3:
            outline = outline[..., 0]
        binary = outline > 0

        # Fill the outline to get a solid cell region
        from scipy.ndimage import binary_fill_holes
        filled = binary_fill_holes(binary)

        if filled.sum() == 0:
            continue

        n_cells += 1
        # Later cells overwrite earlier ones where they overlap (rare)
        instance_mask[filled] = n_cells

    out_path = OUT_GT / f"{stem}.tif"
    tifffile.imwrite(str(out_path), instance_mask)
    print(f"  GT saved: {out_path.name}  ({n_cells} cells)")

    # ── Step 3: copy matching input image (c1 channel = DAPI) ────────────────
    # stem is like "jw-15min 1_c1"; the folder is "jw-15min 1"
    # image filename is "jw-15min 1_c1.TIF"
    folder_name = stem.replace("_c1", "").replace("_c5", "")
    img_path = IMG_ROOT / folder_name / f"{stem}.TIF"

    if not img_path.exists():
        # fallback: try case-insensitive search
        matches = list((IMG_ROOT / folder_name).glob(f"{stem}.*"))
        img_path = matches[0] if matches else None

    if img_path and img_path.exists():
        raw = tifffile.imread(str(img_path))
        # Convert RGB-stored grayscale to true 2D grayscale
        if raw.ndim == 3:
            gray = raw[..., 0].astype(np.float32)
        else:
            gray = raw.astype(np.float32)
        out_img_path = OUT_IMAGES / f"{stem}.tif"
        tifffile.imwrite(str(out_img_path), gray.astype(np.uint8))
        print(f"  IMG saved: {out_img_path.name}")
    else:
        print(f"  WARNING: image not found for stem '{stem}'")

print(f"\nDone. Output in: {OUT_ROOT}")
print(f"  {len(list(OUT_IMAGES.glob('*.tif')))} images")
print(f"  {len(list(OUT_GT.glob('*.tif')))} GT masks")
