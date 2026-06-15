#!/usr/bin/env python3
"""GS2Pano: Render 360° panoramas from 3DGS (CLI with argparse).

Examples:
  # MipNerf360 JSON format, camera 0
  python scripts/render_panorama.py \
    --ply path/to/point_cloud.ply \
    --camera-json path/to/cameras.json \
    --cameras 0 \
    --outdir output

  # COLMAP format, multiple cameras, Mercator
  python scripts/render_panorama.py \
    --ply path/to/point_cloud.ply \
    --colmap path/to/sparse \
    --cameras 0,5,10 \
    --width 1024 --height 512 \
    --projection mercator \
    --outdir output
"""

import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gs2pano.load.ply import load_ply
from gs2pano.load.poses import extract_colmap_poses, extract_json_poses
from gs2pano.render.engine import render, upload_gaussians


def main():
    parser = argparse.ArgumentParser(description="GS2Pano: Render 360° panoramas from 3DGS")
    parser.add_argument("--ply", required=True, help="Path to .ply 3DGS file")
    parser.add_argument("--outdir", required=True, help="Output directory")

    cam_group = parser.add_argument_group("camera source (choose one)")
    cam_group.add_argument("--colmap", help="Path to COLMAP sparse directory")
    cam_group.add_argument("--camera-json", help="Path to cameras.json (MipNerf360 format)")

    parser.add_argument("--cameras", required=True,
                        help="Comma-separated camera indices, e.g. '0,5,10'")
    parser.add_argument("--width", type=int, default=1024, help="Panorama width (default: 1024)")
    parser.add_argument("--height", type=int, default=512, help="Panorama height (default: 512)")
    parser.add_argument("--projection", choices=["equirect", "mercator"],
                        default="equirect", help="Projection type (default: equirect)")

    args = parser.parse_args()

    if args.colmap and args.camera_json:
        parser.error("choose either --colmap or --camera-json, not both")
    if not args.colmap and not args.camera_json:
        parser.error("must specify --colmap or --camera-json")

    camera_ids = [int(x.strip()) for x in args.cameras.split(",")]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load PLY
    print(f"Loading PLY: {args.ply}")
    means_np, scales_np, quats_np, opac_np, colors_np = load_ply(args.ply)
    N = means_np.shape[0]
    print(f"  {N:,} Gaussians")

    # Extract poses
    if args.colmap:
        print(f"Loading COLMAP poses from: {args.colmap}")
        poses = extract_colmap_poses(args.colmap, camera_ids)
    else:
        print(f"Loading camera poses from: {args.camera_json}")
        poses = extract_json_poses(args.camera_json, camera_ids)
    print(f"  {len(poses)} camera(s) ready")

    # Upload Gaussians to GPU
    print("Uploading Gaussians to GPU...")
    gs_data = upload_gaussians(means_np, scales_np, quats_np, opac_np, colors_np)

    # Render each camera
    W, H = args.width, args.height

    for cid in sorted(poses.keys()):
        pose = poses[cid]
        pos, R_c2w, name = pose["pos"], pose["R_c2w"], pose["name"]

        print(f"\nCamera {cid} ({name}): rendering {W}x{H} ({args.projection})...")
        img, dt, mem, n_isects = render(
            means_np, scales_np, quats_np, opac_np, colors_np,
            pos, R_c2w, H, W, gs_data, args.projection,
        )

        nz = (img.sum(axis=-1) > 0.001).sum()
        print(f"  Time: {dt:.1f}s | VRAM: {mem:.0f}MB | "
              f"isects: {n_isects/1e6:.1f}M | NZ: {100*nz/(W*H):.1f}%")

        img_u8 = np.clip(np.clip(img, 0, 1) * 255, 0, 255).astype(np.uint8)
        proj_str = "_mct" if args.projection == "mercator" else "_eqt"
        out_path = outdir / f"panorama_{name.replace('.jpg','')}_{W}x{H}{proj_str}.png"
        Image.fromarray(img_u8).save(str(out_path))
        print(f"  Saved: {out_path} ({os.path.getsize(str(out_path))/1024:.0f} KB)")

    print(f"\nDone. {len(poses)} panorama(s) saved to {outdir}/")


if __name__ == "__main__":
    main()
