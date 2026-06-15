"""GS-ray pairing dataset generation.

Two modes:
  "frustum": record all Gaussians whose spherical-projection pixel bbox
             covers the pixel (no d-squared or T filtering). Large output.
  "ray":     per-ray 3DGUT evaluation with d-squared and T thresholds.
             Compact, matching the rendering kernel's visibility logic.

  In ray mode, the heavy intersection math (hit_t / grayDist / mask) runs
  on GPU via PyTorch, matching the rendering pipeline's computation.
"""

import math
import numba
import numpy as np
import time
import torch
from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

from ..render.projection import spherical_project


# ═══════════════════════════════════════════════════════════════════════════
#  Numba-JIT helper functions (CPU alpha-blending)
# ═══════════════════════════════════════════════════════════════════════════

@numba.njit
def _process_pixel_batch(mask, hit_t, grayDist, gids_arr, opac_arr,
                          d2_thresh, T_thresh):
    """Process all pixels in a tile chunk, returning flat arrays."""
    P, K = mask.shape
    max_pairs = P * K
    out_pi = np.zeros(max_pairs, dtype=np.int32)
    out_gid = np.zeros(max_pairs, dtype=np.int32)
    out_ht = np.zeros(max_pairs, dtype=np.float32)
    out_op = np.zeros(max_pairs, dtype=np.float32)
    out_al = np.zeros(max_pairs, dtype=np.float32)
    out_T = np.zeros(max_pairs, dtype=np.float32)
    n = 0

    for pi in range(P):
        n_vis = 0
        for k in range(K):
            if mask[pi, k]:
                n_vis += 1
        if n_vis == 0:
            continue

        vis_ht = np.zeros(n_vis, dtype=np.float32)
        vis_k = np.zeros(n_vis, dtype=np.int32)
        idx = 0
        for k in range(K):
            if mask[pi, k]:
                vis_ht[idx] = hit_t[pi, k]
                vis_k[idx] = k
                idx += 1

        for i in range(n_vis):
            for j in range(i + 1, n_vis):
                if vis_ht[i] > vis_ht[j]:
                    tmp = vis_ht[i]; vis_ht[i] = vis_ht[j]; vis_ht[j] = tmp
                    tmp = vis_k[i]; vis_k[i] = vis_k[j]; vis_k[j] = tmp

        T = 1.0
        for i in range(n_vis):
            k = vis_k[i]
            gd = grayDist[pi, k]
            resp = math.exp(-0.5 * gd)
            alpha = opac_arr[k] * resp
            if alpha > 0.99:
                alpha = 0.99
            if alpha < 1.0 / 255.0:
                continue

            out_pi[n] = pi
            out_gid[n] = gids_arr[k]
            out_ht[n] = hit_t[pi, k]
            out_op[n] = opac_arr[k]
            out_al[n] = alpha
            out_T[n] = T
            n += 1
            T *= (1.0 - alpha)
            if T < T_thresh:
                break

    return out_pi[:n], out_gid[:n], out_ht[:n], out_op[:n], out_al[:n], out_T[:n]


@numba.njit
def _collect_chunk(mask, hit_t, grayDist, gids_chunk, opac_chunk,
                    out_pi, out_gid, out_ht, out_gd, out_op, offset):
    """Collect visible (pixel, GS) pairs from one chunk into flat arrays."""
    P, C = mask.shape
    n = offset
    for pi in range(P):
        for k in range(C):
            if mask[pi, k]:
                out_pi[n] = pi
                out_gid[n] = gids_chunk[k]
                out_ht[n] = hit_t[pi, k]
                out_gd[n] = grayDist[pi, k]
                out_op[n] = opac_chunk[k]
                n += 1
    return n


@numba.njit
def _process_pixel_batch_sorted(arr_pi, arr_ht, arr_gd, arr_gid, arr_op,
                                  T_thresh):
    """Process hit data grouped by pixel, sorted by hit_t within each group."""
    N = len(arr_pi)
    out_pi = np.zeros(N, dtype=np.int32)
    out_gid = np.zeros(N, dtype=np.int32)
    out_ht = np.zeros(N, dtype=np.float32)
    out_op = np.zeros(N, dtype=np.float32)
    out_al = np.zeros(N, dtype=np.float32)
    out_T = np.zeros(N, dtype=np.float32)
    m = 0

    i = 0
    while i < N:
        T = 1.0
        cur_pi = arr_pi[i]
        while i < N and arr_pi[i] == cur_pi:
            ht = arr_ht[i]; gd = arr_gd[i]; gid = arr_gid[i]; op = arr_op[i]
            resp = math.exp(-0.5 * gd)
            alpha = op * resp
            if alpha > 0.99:
                alpha = 0.99
            if alpha >= 1.0 / 255.0:
                out_pi[m] = cur_pi; out_gid[m] = gid
                out_ht[m] = ht; out_op[m] = op
                out_al[m] = alpha; out_T[m] = T
                m += 1
                T *= (1.0 - alpha)
                if T < T_thresh:
                    while i < N and arr_pi[i] == cur_pi:
                        i += 1
                    break
            i += 1
        while i < N and arr_pi[i] == cur_pi:
            i += 1

    return out_pi[:m], out_gid[:m], out_ht[:m], out_op[:m], out_al[:m], out_T[:m]


# ═══════════════════════════════════════════════════════════════════════════
#  GPU data preparation (called once, shared between render and pair)
# ═══════════════════════════════════════════════════════════════════════════

def _prepare_ray_gpu_data(means_np, scales_np, quats_np, opac_np, cam_pos,
                           R_c2w, H, W, projection):
    """Upload ray-mode data to GPU once. Returns GPU tensors and theta/phi arrays."""
    N = means_np.shape[0]

    # Build inverse-scale * rotation^T for every Gaussian [N, 3, 3]
    S_inv = np.zeros((N, 3, 3), dtype=np.float32)
    S_inv[:, 0, 0] = 1.0 / scales_np[:, 0]
    S_inv[:, 1, 1] = 1.0 / scales_np[:, 1]
    S_inv[:, 2, 2] = 1.0 / scales_np[:, 2]

    wq, xq, yq, zq = quats_np[:, 0], quats_np[:, 1], quats_np[:, 2], quats_np[:, 3]
    Rmat = np.zeros((N, 3, 3), dtype=np.float32)
    Rmat[:, 0, 0] = 1 - 2 * yq * yq - 2 * zq * zq
    Rmat[:, 0, 1] = 2 * xq * yq - 2 * zq * wq
    Rmat[:, 0, 2] = 2 * xq * zq + 2 * yq * wq
    Rmat[:, 1, 0] = 2 * xq * yq + 2 * zq * wq
    Rmat[:, 1, 1] = 1 - 2 * xq * xq - 2 * zq * zq
    Rmat[:, 1, 2] = 2 * yq * zq - 2 * xq * wq
    Rmat[:, 2, 0] = 2 * xq * zq - 2 * yq * wq
    Rmat[:, 2, 1] = 2 * yq * zq + 2 * xq * wq
    Rmat[:, 2, 2] = 1 - 2 * xq * xq - 2 * yq * yq
    iscl_rot_all = S_inv @ Rmat.transpose(0, 2, 1)  # [N, 3, 3]

    # Upload to GPU
    irot_gpu = torch.from_numpy(iscl_rot_all).cuda()
    xyz_gpu = torch.from_numpy(means_np.astype(np.float32)).cuda()
    opac_gpu = torch.from_numpy(opac_np.astype(np.float32)).cuda()
    cam_gpu = torch.from_numpy(cam_pos.astype(np.float32)).cuda()

    # Pre-compute ray directions (vectorized)
    from ..render.rays import _compute_ray_directions
    ray_d_world_all, theta_all, phi_all, _ = _compute_ray_directions(
        H, W, R_c2w, projection)
    ray_d_gpu = torch.from_numpy(ray_d_world_all.reshape(H * W, 3)).cuda()

    return {
        "irot": irot_gpu, "xyz": xyz_gpu, "opac": opac_gpu, "cam": cam_gpu,
        "ray_d": ray_d_gpu, "theta_all": theta_all, "phi_all": phi_all,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Core: paired data generation from pre-computed tile data
# ═══════════════════════════════════════════════════════════════════════════

def generate_from_tiles(means_np, scales_np, quats_np, opac_np,
                         cam_pos, R_c2w, H, W, projection,
                         ug, vg, pr, D_all,
                         isect_offsets_np, flatten_ids_np, n_isects_total,
                         ray_gpu_cache,
                         pair_mode="ray", d2_thresh=9.0, T_thresh=1e-4):
    """Generate GS-ray pairs using pre-computed spherical projection + tile data.

    Shared-path entry point: the caller computes spherical_project + isect_tiles
    once, then calls this alongside the CUDA rasterizer. Eliminates redundancy.

    Returns:
        list of pixel dicts: [{"pixel": [u,v], "theta", "phi", "gaussians": [...]}, ...]
    """
    TS = 16
    tw = math.ceil(W / TS)
    th = math.ceil(H / TS)

    mode_label = "frustum (bbox)" if pair_mode == "frustum" else f"ray (d^2 < {d2_thresh})"
    print(f"  Paired data ({mode_label}): evaluating tiles...")

    if pair_mode == "ray":
        irot_gpu = ray_gpu_cache["irot"]
        xyz_gpu = ray_gpu_cache["xyz"]
        cam_gpu = ray_gpu_cache["cam"]
        ray_d_gpu = ray_gpu_cache["ray_d"]
        theta_all = ray_gpu_cache["theta_all"]
        phi_all = ray_gpu_cache["phi_all"]
        d2_thresh_t = torch.tensor(d2_thresh, device="cuda", dtype=torch.float32)

    pixels_out = []
    total_pairs = 0
    t0 = time.time()
    n_tiles = th * tw
    n_tiles_done = 0
    last_report = 0
    report_interval = max(1, n_tiles // 20)
    n_tiles_with_data = 0

    def _fmt_time(s):
        if s < 60:
            return f"{s:.0f}s"
        m, s = divmod(int(s), 60)
        return f"{m}m{s:02d}s"

    for tv in range(th):
        for tu in range(tw):
            start = int(isect_offsets_np[tv, tu])
            next_tv, next_tu = tv + 1, tu + 1
            if next_tv < th:
                end = int(isect_offsets_np[next_tv, tu])
            elif next_tu < tw:
                end = int(isect_offsets_np[tv, next_tu])
            else:
                end = n_isects_total
            n_tiles_done += 1

            if n_tiles_done - last_report >= report_interval or n_tiles_done == n_tiles:
                last_report = n_tiles_done
                elapsed = time.time() - t0
                pct = 100.0 * n_tiles_done / n_tiles
                eta = elapsed / n_tiles_done * (n_tiles - n_tiles_done) if n_tiles_done > 0 else 0
                print(f"  [paired] {pct:5.1f}% | tiles {n_tiles_done}/{n_tiles} "
                      f"({n_tiles_with_data} non-empty) | {total_pairs} pairs | "
                      f"elapsed {_fmt_time(elapsed)} | ETA {_fmt_time(eta)}",
                      flush=True)

            if start >= end:
                continue
            gids = np.unique(flatten_ids_np[start:end])
            if len(gids) == 0:
                continue
            n_tiles_with_data += 1

            v0, v1 = tv * TS, min(tv * TS + TS, H)
            u0, u1 = tu * TS, min(tu * TS + TS, W)

            if pair_mode == "frustum":
                gs_ug, gs_vg, gs_pr = ug[gids], vg[gids], pr[gids]
                for v in range(v0, v1):
                    for u in range(u0, u1):
                        in_bbox = ((np.abs(gs_ug - u) <= gs_pr) &
                                   (np.abs(gs_vg - v) <= gs_pr))
                        if not in_bbox.any():
                            continue
                        idxs = np.where(in_bbox)[0]
                        gauss_list = [{
                            "idx": int(gids[k]), "t": float(D_all[gids[k]]),
                            "sigma": float(opac_np[gids[k]]),
                            "alpha": 0.0, "T": 0.0,
                        } for k in idxs]
                        pixels_out.append({
                            "pixel": [u, v],
                            "theta": float(2.0 * np.pi * (u + 0.5) / W - np.pi),
                            "phi": float(np.pi * (v + 0.5) / H - np.pi / 2
                                         if projection == "equirect" else
                                         2.0 * np.arctan(np.exp(np.pi * (
                                             2.0 * (v + 0.5) / H - 1.0))) - np.pi / 2),
                            "gaussians": gauss_list,
                        })
                        total_pairs += len(gauss_list)
            else:
                # ---- Ray mode: GPU-accelerated 3DGUT evaluation ----
                K = len(gids)
                P = (v1 - v0) * (u1 - u0)
                row_ids = np.arange(v0, v1)
                col_ids = np.arange(u0, u1)
                pix_flat_ids = (row_ids[:, None] * W + col_ids[None, :]).ravel()
                ray_ds_gpu = ray_d_gpu[torch.from_numpy(pix_flat_ids).long().cuda()]

                gids_t = torch.from_numpy(gids).long().cuda()
                irot_tile = irot_gpu[gids_t]
                xyz_tile = xyz_gpu[gids_t]
                opac_tile_cpu = opac_np[gids]
                pixel_coords = [(v, u) for v in range(v0, v1)
                                for u in range(u0, u1)]

                max_hits = P * K
                all_pi = np.zeros(max_hits, dtype=np.int32)
                all_gid = np.zeros(max_hits, dtype=np.int32)
                all_ht = np.zeros(max_hits, dtype=np.float32)
                all_gd = np.zeros(max_hits, dtype=np.float32)
                all_op = np.zeros(max_hits, dtype=np.float32)
                n_hits = 0

                CHUNK = 4000
                for c_start in range(0, K, CHUNK):
                    c_end = min(c_start + CHUNK, K)
                    gids_chunk = gids[c_start:c_end]

                    xyz_diff = cam_gpu - xyz_tile[c_start:c_end]
                    gro = torch.bmm(irot_tile[c_start:c_end],
                                    xyz_diff.unsqueeze(-1)).squeeze(-1)
                    grd_raw = torch.einsum('kij,pj->pki',
                                           irot_tile[c_start:c_end], ray_ds_gpu)
                    grd_norm = torch.norm(grd_raw, dim=-1, keepdim=True).clamp(min=1e-8)
                    grd = grd_raw / grd_norm
                    gro_brd = gro.unsqueeze(0)
                    hit_t = -torch.sum(grd * gro_brd, dim=-1)
                    gc = torch.cross(grd, gro_brd.expand(P, -1, -1), dim=-1)
                    grayDist = torch.sum(gc * gc, dim=-1)
                    mask = (hit_t > 1e-6) & (grayDist < d2_thresh_t)

                    mask_np = mask.cpu().numpy()
                    n_hits = _collect_chunk(
                        mask_np, hit_t.cpu().numpy(), grayDist.cpu().numpy(),
                        gids_chunk.astype(np.int32),
                        opac_tile_cpu[c_start:c_end].astype(np.float32),
                        all_pi, all_gid, all_ht, all_gd, all_op, n_hits)

                if n_hits == 0:
                    continue

                arr_pi = all_pi[:n_hits]; arr_gid = all_gid[:n_hits]
                arr_ht = all_ht[:n_hits]; arr_gd = all_gd[:n_hits]
                arr_op = all_op[:n_hits]
                sort_idx = np.lexsort((arr_ht, arr_pi))
                arr_pi = arr_pi[sort_idx]; arr_gid = arr_gid[sort_idx]
                arr_ht = arr_ht[sort_idx]; arr_gd = arr_gd[sort_idx]
                arr_op = arr_op[sort_idx]

                out_pi, out_gid, out_ht, out_op, out_al, out_T = (
                    _process_pixel_batch_sorted(
                        arr_pi, arr_ht, arr_gd, arr_gid, arr_op,
                        np.float32(T_thresh)))

                gauss_list = []
                for i in range(len(out_pi)):
                    pi = out_pi[i]
                    v, u = pixel_coords[pi]
                    if i == 0 or out_pi[i] != out_pi[i-1]:
                        if i > 0 and gauss_list:
                            pixels_out.append({
                                "pixel": [prev_u, prev_v],
                                "theta": float(theta_all[prev_u]),
                                "phi": float(phi_all[prev_v]),
                                "gaussians": gauss_list,
                            })
                            total_pairs += len(gauss_list)
                        gauss_list = []
                    gauss_list.append({
                        "idx": int(out_gid[i]), "t": float(out_ht[i]),
                        "sigma": float(out_op[i]),
                        "alpha": float(out_al[i]), "T": float(out_T[i]),
                    })
                    prev_v, prev_u = v, u
                if gauss_list:
                    pixels_out.append({
                        "pixel": [prev_u, prev_v],
                        "theta": float(theta_all[prev_u]),
                        "phi": float(phi_all[prev_v]),
                        "gaussians": gauss_list,
                    })
                    total_pairs += len(gauss_list)

    total_elapsed = time.time() - t0
    print(f"  Paired data complete: {len(pixels_out)} pixels, "
          f"{total_pairs} GS pairs, {_fmt_time(total_elapsed)}")
    return pixels_out


# ═══════════════════════════════════════════════════════════════════════════
#  Standalone wrapper (computes tiles internally)
# ═══════════════════════════════════════════════════════════════════════════

def generate(means_np, scales_np, quats_np, opac_np,
             cam_pos, R_c2w, H, W, gs_data,
             projection="equirect", pair_mode="ray",
             d2_thresh=9.0, T_thresh=1e-4):
    """Standalone GS-ray pairing: computes spherical projection + tiles internally."""
    N = means_np.shape[0]
    TS = 16
    tw = math.ceil(W / TS)
    th = math.ceil(H / TS)

    ug, vg, pr, D_all = spherical_project(
        means_np, scales_np, cam_pos, R_c2w, W, H, projection)

    m2d = torch.zeros(1, N, 2, device="cuda")
    m2d[0, :, 0] = torch.from_numpy(ug.astype(np.float32)).cuda()
    m2d[0, :, 1] = torch.from_numpy(vg.astype(np.float32)).cuda()
    rad = torch.zeros(1, N, 2, device="cuda", dtype=torch.int32)
    rad[0, :, 0] = torch.from_numpy(pr).cuda()
    rad[0, :, 1] = torch.from_numpy(pr).cuda()
    dep = torch.from_numpy(D_all.astype(np.float32)).cuda().unsqueeze(0)
    _, ii, fid = isect_tiles(m2d, rad, dep, TS, tw, th, sort=True)
    iso = isect_offset_encode(ii, 1, tw, th).reshape(1, th, tw)

    isect_offsets_np = iso[0].cpu().numpy()
    flatten_ids_np = fid.cpu().numpy()
    n_isects_total = len(flatten_ids_np)

    if pair_mode == "ray":
        ray_gpu_cache = _prepare_ray_gpu_data(
            means_np, scales_np, quats_np, opac_np, cam_pos,
            R_c2w, H, W, projection)
    else:
        ray_gpu_cache = None

    return generate_from_tiles(
        means_np, scales_np, quats_np, opac_np,
        cam_pos, R_c2w, H, W, projection,
        ug, vg, pr, D_all,
        isect_offsets_np, flatten_ids_np, n_isects_total,
        ray_gpu_cache,
        pair_mode=pair_mode, d2_thresh=d2_thresh, T_thresh=T_thresh)
