#!/usr/bin/env python3
"""Render a 360° panorama from a 3DGS PLY file.

Edit the CONFIG block below, then run:
    python scripts/render_pano.py
"""

import json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gs2pano.load.ply import load_ply
from gs2pano.render.engine import render, upload_gaussians

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
PLY_PATH   = "path/to/point_cloud.ply"
JSON_PATH  = "path/to/cameras.json"
CAM_ID     = 0
OUT_DIR    = "output"
WIDTH      = 1024
HEIGHT     = 512
PROJECTION = "equirect"                # "equirect" or "mercator"
# ══════════════════════════════════════════════════════════════════

OUT_DIR = Path(OUT_DIR); OUT_DIR.mkdir(parents=True, exist_ok=True)
t_total = time.time()

# ── Camera ──
with open(JSON_PATH) as f:
    cams = json.load(f)
cam = cams[CAM_ID]
cam_pos = np.array(cam["position"], dtype=np.float32)
R_c2w   = np.array(cam["rotation"], dtype=np.float32)
cam_name = cam.get("img_name", f"cam{CAM_ID}")
print(f"Camera: {cam_name}  pos={cam_pos}")

# ── Load PLY ──
print(f"Loading PLY: {PLY_PATH}")
means, scales, quats, opacs, colors = load_ply(PLY_PATH)
N = len(means)
print(f"  {N:,} GS")

# ── Upload ──
gs_data = upload_gaussians(means, scales, quats, opacs, colors)

# ── Render ──
print(f"Rendering {WIDTH}x{HEIGHT}...")
img, dt, mem, n_isects = render(
    means, scales, quats, opacs, colors,
    cam_pos, R_c2w, HEIGHT, WIDTH, gs_data, projection=PROJECTION,
)
print(f"  {dt:.2f}s  VRAM: {mem:.0f}MB  isects: {n_isects:,}")

# ── Save ──
img_u8 = np.clip(np.clip(img, 0, 1) * 255, 0, 255).astype(np.uint8)
proj_tag = "mct" if PROJECTION == "mercator" else "eqt"
png_path = OUT_DIR / f"panorama_{cam_name}_{WIDTH}x{HEIGHT}_{proj_tag}.png"
Image.fromarray(img_u8).save(png_path)
nz = (img.sum(axis=-1) > 0.001).sum()
print(f"  PNG: {png_path} ({os.path.getsize(png_path)/1024:.0f}KB)  NZ: {nz}/{WIDTH*HEIGHT}")
print(f"Done in {time.time()-t_total:.1f}s  →  {OUT_DIR}/")
