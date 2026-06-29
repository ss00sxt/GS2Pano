"""Spherical projection for equirectangular and Mercator panoramas.

Maps 3D Gaussian positions to pixel coordinates on the panorama image.
"""

import numpy as np


def spherical_project(means_np, scales_np, cam_pos, R_c2w, W, H,
                      projection="equirect"):
    """Project 3D Gaussians to equirectangular or Mercator pixel space.

    Transforms world-space vectors to camera-local space first, so the
    panorama center (u=W/2) aligns with the camera forward (R_c2w[:,2]).

    Args:
        means_np:    [N, 3] Gaussian centers in world coordinates
        scales_np:   [N, 3] exp-activated scales
        cam_pos:     [3]   camera position in world coordinates
        R_c2w:       [3,3] camera-to-world rotation matrix
        W, H:        panorama width and height in pixels
        projection:  "equirect" or "mercator"

    Returns:
        ug, vg: [N] float pixel coordinates
        pr:     [N] int32 pixel radius (conservative 3-sigma)
        D:      [N] Euclidean distance from camera
    """
    # --- World-to-camera transform ---
    rel_world = means_np - cam_pos
    D = np.linalg.norm(rel_world, axis=1).clip(1e-6)
    rel_cam = rel_world @ R_c2w  # [N, 3] in camera-local space

    # --- Equirectangular angles ---
    theta_g = np.arctan2(rel_cam[:, 0], rel_cam[:, 2])       # azimuth [-pi, pi]
    phi_g   = np.arcsin(np.clip(rel_cam[:, 1] / D, -1, 1))   # elevation [-pi/2, pi/2]

    # --- Azimuth to pixel u (same for both projections) ---
    ug = (theta_g + np.pi) / (2 * np.pi) * W

    # --- Elevation to pixel v ---
    if projection == "mercator":
        # Mercator: psi = ln(tan(pi/4 + phi/2))
        # Clip latitude to ~85 deg to avoid infinite stretching near poles
        phi_clip = np.clip(phi_g, -1.484, 1.484)
        psi = np.log(np.tan(np.pi / 4 + phi_clip / 2))
        vg = (1.0 + psi / np.pi) / 2.0 * H
        # Vertical scale factor: dpsi/dphi = 1/cos(phi)
        v_scale = 1.0 / np.clip(np.cos(phi_clip), 0.087, 1.0)
    else:
        # Equirect: linear mapping from elevation to v
        vg = (phi_g + np.pi / 2) / np.pi * H
        v_scale = 1.0

    # --- Angular extent (3-sigma, conservative) ---
    # At high latitudes, a fixed angular radius covers a wider azimuth span
    # on the equirectangular image: ds^2 = dphi^2 + cos(phi)^2 dtheta^2.
    # Without this factor, tile assignment under-covers the polar bands and
    # produces visible 16x16 tile boundaries.
    sigma_max = np.max(scales_np, axis=1)
    sigma_3 = 3.0 * sigma_max
    angular_r = np.empty_like(D, dtype=np.float32)
    outside = sigma_3 < D
    angular_r[outside] = np.arcsin(np.clip(sigma_3[outside] / D[outside], 0.0, 1.0))
    angular_r[~outside] = np.pi
    u_scale = 1.0 / np.clip(np.cos(phi_g), 1.0 / W, 1.0)
    pix_r_u = np.clip(np.ceil(angular_r / (2 * np.pi) * W * u_scale) + 1, 1, W)
    pix_r_v = np.clip(np.ceil(angular_r / np.pi * H * v_scale) + 1, 1, H)
    pr = np.maximum(pix_r_u, pix_r_v).astype(np.int32)

    return ug, vg, pr, D
