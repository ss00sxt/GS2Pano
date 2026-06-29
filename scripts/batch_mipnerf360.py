#!/usr/bin/env python3
"""Batch render all 9 MipNerf360 scenes: Mercator 1024×512 + paired data."""

import ctypes, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles, rasterize_to_pixels_eval3d_extra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gs2pano.load.ply import load_ply
from gs2pano.render.engine import _build_tile_inputs
from gs2pano.render.projection import spherical_project
from gs2pano.render.rays import generate_rays

# ---- ctypes ----
import gsplat as _gsplat_pkg
_GS_SO = str(Path(_gsplat_pkg.__file__).parent / "csrc.so")
_LIB = ctypes.CDLL(_GS_SO)
_LIB.gs2pano_set_pair_buffers.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_LIB.gs2pano_set_pair_buffers.restype = None
_LIB.gs2pano_set_frustum_data.argtypes = [ctypes.c_void_p]
_LIB.gs2pano_set_frustum_data.restype = None

# ---- Config ----
SRC_DIR    = "path/to/MipNerf360"           # directory with scene subdirs
OUT_DIR    = "path/to/output"               # where to save results
W, H       = 1024, 512
PROJECTION = "mercator"
TS         = 16
PAIR_CAP   = 80_000_000
SCENES     = ["bicycle", "bonsai", "counter", "flowers", "garden",
              "kitchen", "room", "stump", "treehill"]

# ---- Main ----
config = {"resolution": [W, H], "projection": PROJECTION, "scenes": {}}
t_total  = time.time()

for scene in SCENES:
    ply_path = os.path.join(SRC_DIR, scene, "point_cloud.ply")
    cam_path = os.path.join(SRC_DIR, scene, "cameras.json")
    out_dir  = os.path.join(OUT_DIR, scene)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {scene}")
    print(f"{'='*60}")

    # ── Camera ──
    with open(cam_path) as f:
        cams = json.load(f)
    cam = cams[0]
    cam_pos = np.array(cam["position"], dtype=np.float32)
    R_c2w   = np.array(cam["rotation"], dtype=np.float32)
    print(f"  cam0: {cam['img_name']}  pos={cam_pos}")

    # ── Load PLY ──
    means, scales, quats, opacs, colors = load_ply(ply_path)
    N = len(means)
    print(f"  GS: {N:,}")

    # ── Upload ──
    Ms = torch.from_numpy(means).cuda().unsqueeze(0)
    Qs = torch.from_numpy(quats).cuda().unsqueeze(0)
    Ss = torch.from_numpy(scales).cuda().unsqueeze(0)
    Os = torch.from_numpy(opacs).cuda().unsqueeze(0).unsqueeze(1)
    Cs = torch.from_numpy(colors).cuda().unsqueeze(0).unsqueeze(1)
    vm = torch.eye(4, device="cuda").unsqueeze(0).unsqueeze(0)
    Kt = torch.eye(3, device="cuda").unsqueeze(0).unsqueeze(0)

    # ── Projection ──
    tw = math.ceil(W / TS); th = math.ceil(H / TS)
    ug, vg, pr, D = spherical_project(means, scales, cam_pos, R_c2w, W, H, PROJECTION)

    tile_ug, tile_vg, tile_pr, tile_D, tile_ids = _build_tile_inputs(ug, vg, pr, D, W)
    m2d = torch.zeros(1, len(tile_ug), 2, device="cuda")
    rad = torch.zeros(1, len(tile_ug), 2, device="cuda", dtype=torch.int32)
    m2d[0,:,0]=torch.from_numpy(tile_ug).cuda()
    m2d[0,:,1]=torch.from_numpy(tile_vg).cuda()
    rad[0,:,0]=torch.from_numpy(tile_pr).cuda()
    rad[0,:,1]=torch.from_numpy(tile_pr).cuda()
    dep=torch.from_numpy(tile_D).cuda().unsqueeze(0)
    id_map = torch.from_numpy(tile_ids).cuda()
    _, ii, fid = isect_tiles(m2d, rad, dep, TS, tw, th, sort=True)
    fid = id_map[fid]
    iso = isect_offset_encode(ii, 1, tw, th).reshape(1, th, tw)

    # ── Rays + frustum ──
    rays, R_w2c = generate_rays(cam_pos, R_c2w, H, W, PROJECTION)
    fd = np.zeros(12, dtype=np.float32); fd[:9] = R_w2c.ravel()
    _LIB.gs2pano_set_frustum_data(fd.ctypes.data)

    # ── Pair buffers ──
    pair_buf = torch.zeros(PAIR_CAP * 6, dtype=torch.float32, device="cuda")
    pair_cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
    _LIB.gs2pano_set_pair_buffers(pair_buf.data_ptr(), pair_cnt.data_ptr(), PAIR_CAP)

    # ── Render ──
    torch.cuda.synchronize(); t0 = time.time()
    out, _, _, _, _ = rasterize_to_pixels_eval3d_extra(
        means=Ms, quats=Qs, scales=Ss, colors=Cs, opacities=Os,
        viewmats=vm, Ks=Kt, image_width=W, image_height=H, tile_size=TS,
        isect_offsets=iso.cuda(), flatten_ids=fid,
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

    # ── Save PNG ──
    img = out[0].cpu().numpy().reshape(H, W, 3)
    img_u8 = np.clip(np.clip(img, 0, 1) * 255, 0, 255).astype(np.uint8)
    png_path = os.path.join(out_dir, f"{scene}.png")
    Image.fromarray(img_u8).save(png_path)
    nz = (img.sum(axis=-1) > 0.001).sum()

    # ── Sort + pack GS data ──
    t_sort = time.time()
    order = np.lexsort((hit_ts, pix_ids))
    pix_ids = pix_ids[order]; gids = gids[order]
    hit_ts = hit_ts[order]; alphas = alphas[order]
    Ts = Ts[order]; opac_p = opac_p[order]

    P = W * H
    pixel_starts = np.zeros(P + 1, dtype=np.int64)
    counts = np.bincount(pix_ids, minlength=P).astype(np.int64)
    np.cumsum(counts, out=pixel_starts[1:])

    gs_xyz   = means[gids].astype(np.float32)
    gs_rgb   = colors[gids].astype(np.float32)
    gs_scale = scales[gids].astype(np.float32)
    gs_quat  = quats[gids].astype(np.float32)

    from gs2pano.render.rays import _compute_ray_directions
    _, _, _, frustum_bounds = _compute_ray_directions(H, W, R_c2w, PROJECTION)
    pixel_bounds = frustum_bounds.reshape(P, 4)

    # ── Save NPZ ──
    npz_path = os.path.join(out_dir, f"{scene}.npz")
    np.savez_compressed(npz_path,
        pix_id=pix_ids, gid=gids, hit_t=hit_ts, opac=opac_p, alpha=alphas, T=Ts,
        xyz=gs_xyz, rgb=gs_rgb, scale=gs_scale, quat=gs_quat,
        pixel_starts=pixel_starts, pixel_bounds=pixel_bounds,
        cam_pos=cam_pos, W=W, H=H, N_gauss=N, projection=PROJECTION)
    dt_sort = time.time() - t_sort

    n_vis = int((Ts > 0).sum()); n_occ = n - n_vis
    uniq  = len(np.unique(gids))
    n_px  = int((counts > 0).sum())
    dt_total = time.time() - t0 + dt_render

    print(f"  render: {dt_render:.2f}s  sort+pack: {dt_sort:.1f}s  total: {dt_total:.1f}s")
    print(f"  pairs: {n:,} | vis: {n_vis:,} | occ: {n_occ:,} ({100*n_occ/max(1,n):.1f}%)")
    print(f"  unique GS: {uniq:,}/{N:,} ({100*uniq/N:.1f}%) | pixels: {n_px:,} | GS/px: {n/max(1,n_px):.1f}")
    print(f"  PNG: {os.path.getsize(png_path)/1024:.0f}KB | NPZ: {os.path.getsize(npz_path)/1024**2:.0f}MB")
    print(f"  {'⚠️  OVERFLOW!' if overflow else '✅'}")

    config["scenes"][scene] = {
        "camera": cam["img_name"], "n_gauss": N, "n_pairs": n,
        "visible": n_vis, "occluded": n_occ, "unique_gs": uniq,
        "nz_pixels": n_px, "render_s": round(dt_render, 3),
        "total_s": round(dt_total, 1), "overflow": overflow,
    }

    # cleanup
    del Ms, Qs, Ss, Os, Cs, out, pair_buf, pair_cnt, pairs, rays
    torch.cuda.empty_cache()

# ── Save config.json ──
config["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
config["total_elapsed_s"] = round(time.time() - t_total, 1)
config["n_scenes"] = len(config["scenes"])
config["output_dir"] = OUT_DIR
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  ALL DONE in {config['total_elapsed_s']:.0f}s")
print(f"  config.json saved to {OUT_DIR}")
print(f"{'='*60}")
