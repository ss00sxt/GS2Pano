"""Camera pose extraction from COLMAP binary and JSON camera formats.

COLMAP convention (images.bin):
    X_cam = R_w2c * X_world + t          (world-to-camera)
    Camera center:   C = -R_w2c^T * t    (world space)
    Camera-to-world: R_c2w = R_w2c^T

JSON convention 1 (camera_params.json):
    Directly stores camera-to-world (c2w) 4x4 matrices.
    pos = c2w[:3, 3],  R_c2w = c2w[:3, :3]

JSON convention 2 (MipNerf360 / DL3DV cameras.json):
    Stores one dict per camera with `position`, `rotation`, and `img_name`.
    `rotation` is interpreted as camera-to-world.
"""

import json
import os
import struct

import numpy as np


# ─── COLMAP binary reader ──────────────────────────────────────────────

def _read_images_bin(path: str) -> dict:
    """Read COLMAP images.bin, returning {image_id: {q, t, name}}.

    q: [4] float32 quaternion (qw, qx, qy, qz) for world-to-camera rotation
    t: [3] float32 translation for world-to-camera transform
    """
    data = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            tx, ty, tz = struct.unpack("<ddd", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]  # not used, all share one camera
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode()
            npts = struct.unpack("<Q", f.read(8))[0]
            f.seek(npts * 24, 1)  # skip 2D keypoints
            data[iid] = {
                "q": np.array([qw, qx, qy, qz], dtype=np.float32),
                "t": np.array([tx, ty, tz], dtype=np.float32),
                "name": name,
            }
    return data


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (qw, qx, qy, qz) to 3x3 rotation matrix.

    Uses the standard Hamilton product convention (matching GLM / gsplat).
    """
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float32)


# ─── Pose extraction ───────────────────────────────────────────────────

def extract_colmap_poses(sparse_dir: str, camera_ids: list) -> dict:
    """Extract world-space camera poses from a COLMAP sparse directory.

    Args:
        sparse_dir: path to directory containing images.bin
        camera_ids: list of image IDs to extract

    Returns:
        dict: {cid: {"pos": [3], "R_c2w": [3,3], "name": str}}
    """
    images_bin = os.path.join(sparse_dir, "images.bin")
    if not os.path.exists(images_bin):
        raise FileNotFoundError(f"images.bin not found in {sparse_dir}")

    all_images = _read_images_bin(images_bin)
    poses = {}
    for cid in camera_ids:
        if cid not in all_images:
            print(f"  [WARN] camera {cid} not found in images.bin, skipping")
            continue
        img = all_images[cid]
        R_w2c = _quat_to_R(img["q"])
        t = img["t"]
        pos = -R_w2c.T @ t            # world-space camera center
        R_c2w = R_w2c.T               # camera-to-world rotation
        poses[cid] = {"pos": pos, "R_c2w": R_c2w, "name": img["name"]}
    return poses


def extract_json_poses(json_path: str, camera_ids: list) -> dict:
    """Extract world-space camera poses from supported JSON camera formats.

    Args:
        json_path:  path to camera_params.json or cameras.json
        camera_ids: list of camera indices to extract

    Returns:
        dict: {cid: {"pos": [3], "R_c2w": [3,3], "name": str}}
    """
    with open(json_path) as f:
        data = json.load(f)

    poses = {}
    if isinstance(data, list):
        for cid in camera_ids:
            if cid < 0 or cid >= len(data):
                print(f"  [WARN] camera {cid} not found in {json_path}, skipping")
                continue
            cam = data[cid]
            pos = np.array(cam["position"], dtype=np.float32)
            R_c2w = np.array(cam["rotation"], dtype=np.float32)
            name = cam.get("img_name", f"cam_{cam.get('id', cid)}")
            poses[cid] = {"pos": pos, "R_c2w": R_c2w, "name": name}
        return poses

    for cid in camera_ids:
        for ext in data["extrinsics"]:
            if ext["camera_id"] == cid:
                c2w = np.array(ext["matrix"], dtype=np.float32)
                pos = c2w[:3, 3]               # camera position
                R_c2w = c2w[:3, :3]             # camera-to-world rotation
                poses[cid] = {"pos": pos, "R_c2w": R_c2w, "name": f"cam_{cid}"}
                break
        if cid not in poses:
            print(f"  [WARN] camera {cid} not found in {json_path}, skipping")
    return poses
