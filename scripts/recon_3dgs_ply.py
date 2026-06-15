#!/usr/bin/env python3
"""Reconstruct a 3DGS PLY file from a .npz paired dataset.

Extracts all unique Gaussians referenced in the npz and copies their
attributes (position, SH, opacity, scale, rotation) from the original
PLY.  The output is a binary PLY viewable in SuperSplat.

Usage:
    python scripts/recon_3dgs_ply.py \
        --npz output/scene/panorama_xxx.npz \
        --ply path/to/original.ply \
        --out reconstructed.ply
"""

import argparse, sys
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Reconstruct PLY from paired-data npz")
    parser.add_argument("--npz", required=True, help="Path to .npz paired dataset")
    parser.add_argument("--ply", required=True, help="Path to original 3DGS .ply")
    parser.add_argument("--out", required=True, help="Output PLY path")
    parser.add_argument("--visible-only", action="store_true",
                        help="Only include visible (T>0) GS")
    args = parser.parse_args()

    print(f"Loading pairs: {args.npz}")
    data = np.load(args.npz)
    gids = data["gid"]
    Ts = data["T"]

    if args.visible_only:
        mask = Ts > 0
        unique_gids = np.unique(gids[mask])
        label = "visible"
    else:
        unique_gids = np.unique(gids)
        label = "all"

    print(f"  {len(unique_gids):,} unique GS ({label}), from {len(gids):,} total pairs")

    print(f"Loading original PLY: {args.ply}")
    ply = PlyData.read(args.ply)
    v = ply["vertex"]
    N = v.count
    keep = np.sort(unique_gids)
    M = len(keep)

    vertex_data = np.zeros(M, dtype=v.data.dtype)
    for name in v.data.dtype.names:
        vertex_data[name] = v[name][keep]

    print(f"  Extracted {M:,} GS ({100*M/N:.1f}% of original {N:,})")

    print(f"Writing: {args.out}")
    el = PlyElement.describe(vertex_data, "vertex")
    PlyData([el], text=False).write(args.out)

    import os
    size_mb = os.path.getsize(args.out) / 1024**2
    print(f"  Saved: {args.out} ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
