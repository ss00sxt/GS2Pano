#!/usr/bin/env python3
"""Render a 360° panorama + generate GS-ray paired dataset.

Edit the CONFIG block below, then run:
    python scripts/render_and_pair.py

Output:
  - panorama_{name}_{W}x{H}_{proj}.png   rendered panorama
  - panorama_{name}_{W}x{H}_{proj}.npz   paired dataset
  - panorama_{name}_{W}x{H}_{proj}.ply   reconstructed GS (optional)
"""

import ctypes, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles, rasterize_to_pixels_eval3d_extra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gs2pano.load.ply import load_ply
from gs2pano.load.poses import extract_colmap_poses
from gs2pano.render.projection import spherical_project
from gs2pano.render.rays import generate_rays, _compute_ray_directions

# ---- ctypes setup ----
import gsplat as _gsplat_pkg
_GS_SO = str(Path(_gsplat_pkg.__file__).parent / "csrc.so")
_LIB = ctypes.CDLL(_GS_SO)
_LIB.gs2pano_set_pair_buffers.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_LIB.gs2pano_set_pair_buffers.restype = None
_LIB.gs2pano_set_frustum_data.argtypes = [ctypes.c_void_p]
_LIB.gs2pano_set_frustum_data.restype = None

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
PLY_PATH   = "path/to/point_cloud.ply"
JSON_PATH  = "path/to/cameras.json"  # MipNerf360 format
CAM_ID     = 0                       # camera index
OUT_DIR    = "output"
WIDTH      = 1024
HEIGHT     = 512
PROJECTION = "mercator"              # "equirect" or "mercator"
PAIR_CAP   = 80_000_000              # GPU buffer capacity (pairs)
GEN_PLY    = True                    # also reconstruct a PLY?
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
fd = np.zeros(12, dtype=np.float32); fd[:9] = R_w2c.ravel()
_LIB.gs2pano_set_frustum_data(fd.ctypes.data)

# ── Pair buffers ──
pair_buf = torch.zeros(PAIR_CAP * 6, dtype=torch.float32, device="cuda")
pair_cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
_LIB.gs2pano_set_pair_buffers(pair_buf.data_ptr(), pair_cnt.data_ptr(), PAIR_CAP)

# ── Render + collect pairs ──
print(f"Rendering {WIDTH}x{HEIGHT} + pairs...")
torch.cuda.synchronize(); t0 = time.time()
out, _, _, _, _ = rasterize_to_pixels_eval3d_extra(
    means=Ms, quats=Qs, scales=Ss, colors=Cs, opacities=Os,
    viewmats=vm, Ks=Kt, image_width=WIDTH, image_height=HEIGHT,
    tile_size=TS, isect_offsets=iso.cuda(), flatten_ids=fid,
    backgrounds=None, camera_model="pinhole", rays=rays,
)
torch.cuda.synchronize()
dt_render = time.time() - t0

# ── Read pairs ──
n_pairs = int(pair_cnt.cpu().item())
_LIB.gs2pano_set_pair_buffers(0, 0, 0)
overflow = n_pairs >= PAIR_CAP
n = min(n_pairs, PAIR_CAP)
pairs = pair_buf[:n*6].cpu().numpy().reshape(n, 6)
pix_ids = pairs[:,0].astype(np.int32); gids = pairs[:,1].astype(np.int32)
hit_ts  = pairs[:,2]; alphas = pairs[:,4]; Ts = pairs[:,5]; opac_p = pairs[:,3]
print(f"  render: {dt_render:.2f}s  pairs: {n:,}" + (" OVERFLOW!" if overflow else ""))

# ── Save PNG ──
img = out[0].cpu().numpy().reshape(HEIGHT, WIDTH, 3)
img_u8 = np.clip(np.clip(img, 0, 1) * 255, 0, 255).astype(np.uint8)
proj_tag = "mct" if PROJECTION == "mercator" else "eqt"
png_name = f"panorama_{cam_name}_{WIDTH}x{HEIGHT}_{proj_tag}.png"
png_path = OUT_DIR / png_name
Image.fromarray(img_u8).save(png_path)
nz_px = (img.sum(axis=-1) > 0.001).sum()
print(f"  PNG: {png_path} ({os.path.getsize(png_path)/1024:.0f}KB)  NZ: {nz_px}/{WIDTH*HEIGHT}")

# ── Sort + pack ──
print("Sorting + packing GS data...")
order = np.lexsort((hit_ts, pix_ids))
pix_ids = pix_ids[order]; gids = gids[order]; hit_ts = hit_ts[order]
alphas = alphas[order]; Ts = Ts[order]; opac_p = opac_p[order]

P = WIDTH * HEIGHT
pixel_starts = np.zeros(P + 1, dtype=np.int64)
counts = np.bincount(pix_ids, minlength=P).astype(np.int64)
np.cumsum(counts, out=pixel_starts[1:])

gs_xyz   = means[gids].astype(np.float32)
gs_rgb   = colors[gids].astype(np.float32)
gs_scale = scales[gids].astype(np.float32)
gs_quat  = quats[gids].astype(np.float32)

_, _, _, frustum_bounds = _compute_ray_directions(HEIGHT, WIDTH, R_c2w, PROJECTION)
pixel_bounds = frustum_bounds.reshape(P, 4)

# ── Save NPZ ──
npz_path = OUT_DIR / png_name.replace(".png", ".npz")
np.savez_compressed(npz_path,
    pix_id=pix_ids, gid=gids, hit_t=hit_ts, opac=opac_p, alpha=alphas, T=Ts,
    xyz=gs_xyz, rgb=gs_rgb, scale=gs_scale, quat=gs_quat,
    pixel_starts=pixel_starts, pixel_bounds=pixel_bounds,
    cam_pos=cam_pos, W=WIDTH, H=HEIGHT, N_gauss=N, projection=PROJECTION)
print(f"  NPZ: {npz_path} ({os.path.getsize(npz_path)/1024**2:.0f}MB)")

# ── Stats ──
n_vis = int((Ts > 0).sum()); n_occ = n - n_vis
uniq = len(np.unique(gids)); n_px = int((counts > 0).sum())
print(f"  {n:,} pairs | {n_vis:,} visible | {n_occ:,} occluded ({100*n_occ/max(1,n):.1f}%)")
print(f"  {uniq:,} unique GS / {N:,} ({100*uniq/N:.1f}%) | {n_px:,} pixels | {n/max(1,n_px):.1f} GS/px")

# ── Reconstruct PLY ──
if GEN_PLY:
    print("Reconstructing PLY...")
    ply = PlyData.read(PLY_PATH)
    keep = np.sort(np.unique(gids))
    vdata = np.zeros(len(keep), dtype=ply["vertex"].data.dtype)
    for name in ply["vertex"].data.dtype.names:
        vdata[name] = ply["vertex"][name][keep]
    el = PlyElement.describe(vdata, "vertex")
    ply_out = OUT_DIR / png_name.replace(".png", ".ply")
    PlyData([el], text=False).write(ply_out)
    print(f"  PLY: {ply_out} ({os.path.getsize(ply_out)/1024**2:.0f}MB, {len(keep):,} GS)")

print(f"\nDone in {time.time()-t_total:.1f}s  →  {OUT_DIR}/")
