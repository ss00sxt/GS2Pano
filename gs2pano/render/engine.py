"""Core panorama rendering pipeline.

Pipeline:
  upload Gaussians to GPU -> spherical projection -> tile assignment
  -> generate rays -> CUDA 3DGUT rasterization
"""

import ctypes
import math
import time
from pathlib import Path

import numpy as np
import torch
from gsplat.cuda._wrapper import (
    isect_offset_encode, isect_tiles,
    rasterize_to_pixels_eval3d_extra,
)

from .projection import spherical_project
from .rays import generate_rays

# ---- ctypes: set frustum params (R_w2c) on GPU ----
_GS_SO = None
_LIB = None


def _get_lib():
    global _GS_SO, _LIB
    if _LIB is None:
        import gsplat as _gsplat_pkg
        _GS_SO = str(Path(_gsplat_pkg.__file__).parent / "csrc.so")
        _LIB = ctypes.CDLL(_GS_SO)
        _LIB.gs2pano_set_frustum_data.argtypes = [ctypes.c_void_p]
        _LIB.gs2pano_set_frustum_data.restype = None
    return _LIB


def _set_frustum_params(R_w2c):
    """Upload R_w2c (world-to-camera rotation) to GPU global memory.

    IMPORTANT: cudaMemcpyToSymbol src must be a HOST pointer, not device.
    """
    data = np.zeros(12, dtype=np.float32)
    data[:9] = R_w2c.ravel()
    _get_lib().gs2pano_set_frustum_data(data.ctypes.data)


def upload_gaussians(means_np, scales_np, quats_np, opac_np, colors_np):
    """Upload Gaussian data to GPU and wrap with batch/camera dimensions.

    All inputs are [N, ...] numpy arrays. Returns GPU tensors with shapes
    suitable for rasterize_to_pixels_eval3d_extra (batch_dim + camera_dim).

    Returns:
        (Ms, Qs, Ss, Os, Cs, vm, Kt) — all on GPU.
    """
    Ms = torch.from_numpy(means_np).cuda().unsqueeze(0)          # [1, N, 3]
    Qs = torch.from_numpy(quats_np).cuda().unsqueeze(0)          # [1, N, 4]
    Ss = torch.from_numpy(scales_np).cuda().unsqueeze(0)         # [1, N, 3]
    Os = torch.from_numpy(opac_np).cuda().unsqueeze(0).unsqueeze(1)     # [1, 1, N]
    Cs = torch.from_numpy(colors_np).cuda().unsqueeze(0).unsqueeze(1)   # [1, 1, N, 3]
    # Identity viewmat and intrinsics (per-ray mode ignores these)
    vm = torch.eye(4, device="cuda").unsqueeze(0).unsqueeze(0)   # [1, 1, 4, 4]
    Kt = torch.eye(3, device="cuda").unsqueeze(0).unsqueeze(0)   # [1, 1, 3, 3]
    return Ms, Qs, Ss, Os, Cs, vm, Kt


def render(means_np, scales_np, quats_np, opac_np, colors_np,
           cam_pos, R_c2w, H, W, gs_data, projection="equirect"):
    """Render a single equirectangular or Mercator panorama.

    Args:
        means_np / scales_np / quats_np / opac_np / colors_np:
            [N, ...] float32 numpy arrays for Gaussian parameters.
        cam_pos:     [3]   world-space camera position
        R_c2w:       [3,3] camera-to-world rotation matrix
        H, W:        output resolution in pixels
        gs_data:     tuple from upload_gaussians()
        projection:  "equirect" or "mercator"

    Returns:
        img:       [H, W, 3] HDR numpy array (not clipped to [0,1])
        dt:        render time in seconds
        mem_mb:    peak GPU memory in MB
        n_isects:  number of tile-Gaussian intersections
    """
    Ms, Qs, Ss, Os, Cs, vm, Kt = gs_data
    N = means_np.shape[0]
    TS = 16  # CUDA tile size
    tw = math.ceil(W / TS)
    th = math.ceil(H / TS)

    # ---- Phase 1': spherical projection ----
    # Map each 3D Gaussian to its pixel coordinate and radius on the panorama
    ug, vg, pr, D = spherical_project(
        means_np, scales_np, cam_pos, R_c2w, W, H, projection)

    # ---- Phase 3: tile assignment ----
    # Build CUDA tensors for gsplat's isect_tiles (expects [1, N, 2] etc.)
    m2d = torch.zeros(1, N, 2, device="cuda")
    m2d[0, :, 0] = torch.from_numpy(ug.astype(np.float32)).cuda()
    m2d[0, :, 1] = torch.from_numpy(vg.astype(np.float32)).cuda()
    rad = torch.zeros(1, N, 2, device="cuda", dtype=torch.int32)
    rad[0, :, 0] = torch.from_numpy(pr).cuda()
    rad[0, :, 1] = torch.from_numpy(pr).cuda()
    dep = torch.from_numpy(D.astype(np.float32)).cuda().unsqueeze(0)

    # isect_tiles: assign each Gaussian to all tiles its pixel bbox overlaps
    _, ii, fid = isect_tiles(m2d, rad, dep, TS, tw, th, sort=True)
    # isect_offset_encode: build per-tile index into flatten_ids
    iso = isect_offset_encode(ii, 1, tw, th).reshape(1, th, tw)

    # ---- Generate per-pixel rays ----
    rays, R_w2c = generate_rays(cam_pos, R_c2w, H, W, projection)
    _set_frustum_params(R_w2c)

    # ---- Phase 4: CUDA 3DGUT rasterization ----
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.time()

    out, _, _, _, _ = rasterize_to_pixels_eval3d_extra(
        means=Ms, quats=Qs, scales=Ss, colors=Cs, opacities=Os,
        viewmats=vm, Ks=Kt, image_width=W, image_height=H, tile_size=TS,
        isect_offsets=iso.cuda(), flatten_ids=fid,
        backgrounds=None, camera_model="pinhole", rays=rays,
    )

    torch.cuda.synchronize()
    dt = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    img = out[0].cpu().numpy().reshape(H, W, 3)
    return img, dt, mem_mb, ii.numel()


def render_with_pairs(means_np, scales_np, quats_np, opac_np, colors_np,
                      cam_pos, R_c2w, H, W, gs_data, projection="equirect",
                      pair_mode="ray", d2_thresh=9.0, T_thresh=1e-4):
    """Render a panorama AND generate paired data in one pass.

    Spherical projection and tile assignment are computed once and shared
    between the CUDA rasterizer and the GPU-accelerated pairing logic.

    Returns:
        img:       [H, W, 3] HDR numpy array
        dt:        render time in seconds
        mem_mb:    peak GPU memory in MB
        n_isects:  number of tile-Gaussian intersections
        pixels:    list of pixel dicts (paired data), or None if pair_mode=None
    """
    from ..output.paired_data import generate_from_tiles, _prepare_ray_gpu_data

    Ms, Qs, Ss, Os, Cs, vm, Kt = gs_data
    N = means_np.shape[0]
    TS = 16
    tw = math.ceil(W / TS)
    th = math.ceil(H / TS)

    # ---- Phase 1': spherical projection (shared) ----
    from .projection import spherical_project
    ug, vg, pr, D = spherical_project(
        means_np, scales_np, cam_pos, R_c2w, W, H, projection)

    # ---- Phase 2: prepare ray GPU data (if pairing) ----
    if pair_mode is not None:
        ray_gpu_cache = _prepare_ray_gpu_data(
            means_np, scales_np, quats_np, opac_np, cam_pos,
            R_c2w, H, W, projection)
    else:
        ray_gpu_cache = None

    # ---- Phase 3: tile assignment (shared) ----
    m2d = torch.zeros(1, N, 2, device="cuda")
    m2d[0, :, 0] = torch.from_numpy(ug.astype(np.float32)).cuda()
    m2d[0, :, 1] = torch.from_numpy(vg.astype(np.float32)).cuda()
    rad = torch.zeros(1, N, 2, device="cuda", dtype=torch.int32)
    rad[0, :, 0] = torch.from_numpy(pr).cuda()
    rad[0, :, 1] = torch.from_numpy(pr).cuda()
    dep = torch.from_numpy(D.astype(np.float32)).cuda().unsqueeze(0)

    _, ii, fid = isect_tiles(m2d, rad, dep, TS, tw, th, sort=True)
    iso = isect_offset_encode(ii, 1, tw, th).reshape(1, th, tw)

    # ---- Phase 4: generate rays + CUDA rasterization (render) ----
    from .rays import generate_rays
    rays, R_w2c = generate_rays(cam_pos, R_c2w, H, W, projection)
    _set_frustum_params(R_w2c)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.time()

    out, _, _, _, _ = rasterize_to_pixels_eval3d_extra(
        means=Ms, quats=Qs, scales=Ss, colors=Cs, opacities=Os,
        viewmats=vm, Ks=Kt, image_width=W, image_height=H, tile_size=TS,
        isect_offsets=iso.cuda(), flatten_ids=fid,
        backgrounds=None, camera_model="pinhole", rays=rays,
    )

    torch.cuda.synchronize()
    dt = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    img = out[0].cpu().numpy().reshape(H, W, 3)

    # ---- Phase 5: paired data generation (reuses same tile data) ----
    if pair_mode is not None:
        isect_offsets_np = iso[0].cpu().numpy()
        flatten_ids_np = fid.cpu().numpy()
        pixels = generate_from_tiles(
            means_np, scales_np, quats_np, opac_np,
            cam_pos, R_c2w, H, W, projection,
            ug, vg, pr, D,
            isect_offsets_np, flatten_ids_np, len(flatten_ids_np),
            ray_gpu_cache,
            pair_mode=pair_mode, d2_thresh=d2_thresh, T_thresh=T_thresh)
    else:
        pixels = None

    return img, dt, mem_mb, ii.numel(), pixels
