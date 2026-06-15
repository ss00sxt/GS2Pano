"""Load a 3DGS .ply file into numpy arrays for rendering.

Handles both standard inria-format PLY files (opacity already in [0,1])
and raw-logit opacity (e.g., mkbgs dataset). Auto-detects by checking
if max(opacity) > 1.0.
"""

import numpy as np
from plyfile import PlyData


def load_ply(ply_path: str):
    """Load 3DGS PLY, returning (means, scales, quats, opacities, colors).

    All outputs are float32 numpy arrays:
        means:     [N, 3]  Gaussian centers in world coordinates
        scales:    [N, 3]  exp-activated scales (exp of stored log-scale)
        quats:     [N, 4]  normalized quaternions (w, x, y, z)
        opacities: [N]     sigmoid-activated opacity in [0, 1]
        colors:    [N, 3]  SH DC colors in [0, inf) (HDR, not clipped)
    """
    ply = PlyData.read(ply_path)
    v = ply["vertex"]

    # --- Positions ---
    means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    # --- Scales (exp of stored log-scale) ---
    scales = np.exp(np.stack(
        [v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1
    ).astype(np.float32))

    # --- Quaternions (normalize to unit) ---
    quats = np.stack(
        [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1
    ).astype(np.float32)
    quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)

    # --- Opacity (auto-detect raw logit vs sigmoid) ---
    opacities = v["opacity"].astype(np.float32)
    if opacities.max() > 1.0:
        # Raw logit stored in PLY — apply sigmoid activation
        opacities = 1.0 / (1.0 + np.exp(-opacities))

    # --- Color from SH DC (degree-0) coefficients ---
    # gsplat formula: color = clamp(spherical_harmonics() + 0.5, 0, inf)
    # DC-only: color = 0.5 + SH_C0 * f_dc
    sh_c0 = 0.28209479177387814
    colors = np.stack(
        [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1
    ).astype(np.float32)
    colors = np.clip(0.5 + sh_c0 * colors, 0.0, None)

    return means, scales, quats, opacities, colors
