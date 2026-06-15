#!/usr/bin/env python3
"""Render a 360° panorama from a 3DGS PLY file.

Edit the CONFIG block below, then run:
    python scripts/render_pano.py
"""

import ctypes, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles, rasterize_to_pixels_eval3d_extra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gs2pano.load.ply import load_ply
from gs2pano.render.projection import spherical_project
from gs2pano.render.rays import generate_rays

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
PLY_PATH   = "path/to/point_cloud.ply"
JSON_PATH  = "path/to/cameras.json"
CAM_ID     = 0
OUT_DIR    = "output"
WIDTH      = 1024
HEIGHT     = 512
PROJECTION = "mercator"                # "equirect" or "mercator"
# ══════════════════════════════════════════════════════════════════

TS = 16
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
Ms = torch.from_numpy(means).cuda().unsqueeze(0)
Qs = torch.from_numpy(quats).cuda().unsqueeze(0)
Ss = torch.from_numpy(scales).cuda().unsqueeze(0)
Os = torch.from_numpy(opacs).cuda().unsqueeze(0).unsqueeze(1)
Cs = torch.from_numpy(colors).cuda().unsqueeze(0).unsqueeze(1)
vm = torch.eye(4, device="cuda").unsqueeze(0).unsqueeze(0)
Kt = torch.eye(3, device="cuda").unsqueeze(0).unsqueeze(0)

# ── Projection + Tiles ──
tw = math.ceil(WIDTH / TS); th = math.ceil(HEIGHT / TS)
ug, vg, pr, D = spherical_project(means, scales, cam_pos, R_c2w, WIDTH, HEIGHT, PROJECTION)
print(f"Proj: pr max={pr.max()} mean={pr.mean():.1f}")

m2d = torch.zeros(1, N, 2, device="cuda")
rad = torch.zeros(1, N, 2, device="cuda", dtype=torch.int32)
m2d[0,:,0]=torch.from_numpy(ug.astype(np.float32)).cuda()
m2d[0,:,1]=torch.from_numpy(vg.astype(np.float32)).cuda()
rad[0,:,0]=torch.from_numpy(pr).cuda(); rad[0,:,1]=torch.from_numpy(pr).cuda()
dep=torch.from_numpy(D.astype(np.float32)).cuda().unsqueeze(0)
_, ii, fid = isect_tiles(m2d, rad, dep, TS, tw, th, sort=True)
iso = isect_offset_encode(ii, 1, tw, th).reshape(1, th, tw)
print(f"Tiles: {fid.numel():,} isects")

# ── Rays + frustum ──
rays, R_w2c = generate_rays(cam_pos, R_c2w, HEIGHT, WIDTH, PROJECTION)
import gsplat
_lib = ctypes.CDLL(str(Path(gsplat.__file__).parent / "csrc.so"))
_lib.gs2pano_set_frustum_data.argtypes = [ctypes.c_void_p]
_lib.gs2pano_set_frustum_data.restype = None
fd = np.zeros(12, dtype=np.float32); fd[:9] = R_w2c.ravel()
_lib.gs2pano_set_frustum_data(fd.ctypes.data)

# ── Render ──
print(f"Rendering {WIDTH}x{HEIGHT}...")
torch.cuda.synchronize(); t0 = time.time()
out, _, _, _, _ = rasterize_to_pixels_eval3d_extra(
    means=Ms, quats=Qs, scales=Ss, colors=Cs, opacities=Os,
    viewmats=vm, Ks=Kt, image_width=WIDTH, image_height=HEIGHT,
    tile_size=TS, isect_offsets=iso.cuda(), flatten_ids=fid,
    backgrounds=None, camera_model="pinhole", rays=rays,
)
torch.cuda.synchronize()
dt = time.time() - t0; mem = torch.cuda.max_memory_allocated() / 1024**2
print(f"  {dt:.2f}s  VRAM: {mem:.0f}MB")

# ── Save ──
img = out[0].cpu().numpy().reshape(HEIGHT, WIDTH, 3)
img_u8 = np.clip(np.clip(img, 0, 1) * 255, 0, 255).astype(np.uint8)
proj_tag = "mct" if PROJECTION == "mercator" else "eqt"
png_path = OUT_DIR / f"panorama_{cam_name}_{WIDTH}x{HEIGHT}_{proj_tag}.png"
Image.fromarray(img_u8).save(png_path)
nz = (img.sum(axis=-1) > 0.001).sum()
print(f"  PNG: {png_path} ({os.path.getsize(png_path)/1024:.0f}KB)  NZ: {nz}/{WIDTH*HEIGHT}")
print(f"Done in {time.time()-t_total:.1f}s  →  {OUT_DIR}/")
