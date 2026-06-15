"""Generate per-pixel world-space rays for equirectangular and Mercator panoramas.

Camera-local coordinate convention: X=right, Y=up, Z=forward (std math).
"""

import math
import sys
import numpy as np
import torch


def _compute_ray_directions(H, W, R_c2w, projection="equirect"):
    """Vectorized: compute world-space ray directions for all pixels at once.

    Returns:
        ray_d_world:   [H, W, 3] float32 numpy array of unit directions
        theta_all:     [W] float32 azimuth per column
        phi_all:       [H] float32 elevation per row
        frustum_bounds:[H, W, 4] float32 (theta_min, theta_max, phi_min, phi_max)
    """
    # Azimuth per column: [W]
    u_coords = np.arange(W, dtype=np.float32)
    theta_all = (2.0 * np.pi * (u_coords + 0.5) / W - np.pi).astype(np.float32)

    # Elevation per row: [H]
    v_coords = np.arange(H, dtype=np.float32)
    if projection == "mercator":
        psi_all = np.pi * (2.0 * (v_coords + 0.5) / H - 1.0)
        phi_all = (2.0 * np.arctan(np.exp(psi_all)) - np.pi / 2).astype(np.float32)
    else:
        phi_all = (np.pi * (v_coords + 0.5) / H - np.pi / 2).astype(np.float32)

    # Camera-local directions via broadcasting: [H,1,3] * [1,W,3]
    cp = np.cos(phi_all, dtype=np.float32)  # [H]
    sp = np.sin(phi_all, dtype=np.float32)  # [H]
    ct = np.cos(theta_all, dtype=np.float32)  # [W]
    st = np.sin(theta_all, dtype=np.float32)  # [W]

    # local_dir = (cos(phi)*sin(theta), sin(phi), cos(phi)*cos(theta))
    local_x = cp[:, None] * st[None, :]  # [H, W]
    local_y = np.tile(sp[:, None], (1, W))  # [H, W]
    local_z = cp[:, None] * ct[None, :]  # [H, W]
    local = np.stack([local_x, local_y, local_z], axis=-1)  # [H, W, 3]

    # Rotate to world space: [H, W, 3] -> [H, W, 3]
    # local[i,j] is a row vector; world = local @ R_c2w^T
    R = R_c2w.astype(np.float32).T  # [3, 3] to multiply row vectors
    ray_d_world = local @ R  # [H, W, 3]

    # Normalize
    norm = np.linalg.norm(ray_d_world, axis=-1, keepdims=True)
    ray_d_world = ray_d_world / np.maximum(norm, 1e-8)

    # ---- Frustum bounds: angular extent of each pixel ----
    # theta bounds: W+1 edges along azimuth, covering [-pi, pi]
    theta_edges = np.linspace(-np.pi, np.pi, W + 1, dtype=np.float32)
    theta_min = theta_edges[:-1]      # [W], per-column lower bound
    theta_max = theta_edges[1:]       # [W], per-column upper bound

    # phi bounds depend on projection
    if projection == "mercator":
        # Mercator: compute phi at pixel edges (v + 0, v + 1) via inverse
        v_edges = np.arange(H + 1, dtype=np.float32)
        psi_edges = np.pi * (2.0 * v_edges / H - 1.0)
        phi_edges = 2.0 * np.arctan(np.exp(psi_edges)) - np.pi / 2
        phi_min = phi_edges[:-1]      # [H], per-row lower bound
        phi_max = phi_edges[1:]       # [H], per-row upper bound
    else:
        # Equirect: uniform in phi
        phi_edges = np.linspace(-np.pi / 2, np.pi / 2, H + 1, dtype=np.float32)
        phi_min = phi_edges[:-1]
        phi_max = phi_edges[1:]

    # Broadcast to per-pixel: [H, W, 4]
    frustum_bounds = np.zeros((H, W, 4), dtype=np.float32)
    frustum_bounds[:, :, 0] = theta_min[None, :]   # theta_min
    frustum_bounds[:, :, 1] = theta_max[None, :]   # theta_max
    frustum_bounds[:, :, 2] = phi_min[:, None]     # phi_min
    frustum_bounds[:, :, 3] = phi_max[:, None]     # phi_max

    return ray_d_world, theta_all, phi_all, frustum_bounds


def generate_rays(cam_pos, R_c2w, H, W, projection="equirect"):
    """Generate per-pixel world-space rays + frustum bounds for a panorama.

    Returns:
        rays:   [1, 1, H*W, 10] GPU tensor
                last dim = (ox, oy, oz, dx, dy, dz,
                            theta_min, theta_max, phi_min, phi_max)
        R_w2c:  [3, 3] world-to-camera rotation (for kernel frustum calc)
    """
    # Vectorized ray direction computation
    ray_d_world, _, _, frustum_bounds = _compute_ray_directions(
        H, W, R_c2w, projection)

    # Build rays tensor: origins + directions + frustum bounds
    origins = np.tile(cam_pos.astype(np.float32).reshape(1, 1, 3), (H, W, 1))
    rays_np = np.concatenate([origins, ray_d_world, frustum_bounds], axis=-1)  # [H, W, 10]
    rays_np = rays_np.reshape(1, 1, H * W, 10)

    rays = torch.from_numpy(rays_np).cuda()
    R_w2c = R_c2w.T.astype(np.float32)
    return rays, R_w2c
