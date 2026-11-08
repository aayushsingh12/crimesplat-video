"""Training views: undistort registered keyframes and their masks to a shared pinhole camera.

Undistortion is done with our own remap (via pycolmap's camera model) so images and masks use
exactly the same mapping, for any COLMAP model including OPENCV_FISHEYE. Pixels that fall
outside the source image are added to the mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap


@dataclass
class View:
    name: str
    image: np.ndarray  # HxWx3 uint8, RGB, undistorted
    mask: np.ndarray  # HxW bool, True = ignore
    K: np.ndarray  # 3x3 pinhole intrinsics at this resolution
    w2c: np.ndarray  # 4x4 world-to-camera, OpenCV convention


def _pinhole_remap(cam: pycolmap.Camera, long_side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    s = min(1.0, long_side / max(cam.width, cam.height))
    w, h = round(cam.width * s), round(cam.height * s)
    f = cam.mean_focal_length() * s
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack([(u - K[0, 2]) / f, (v - K[1, 2]) / f, np.ones_like(u)], -1).reshape(-1, 3)
    src = cam.img_from_cam(rays).reshape(h, w, 2) - 0.5
    return src[..., 0].astype(np.float32), src[..., 1].astype(np.float32), K, w, h


def load_views(sparse: Path, img_dir: Path, mask_dir: Path | None, long_side: int) -> list[View]:
    rec = pycolmap.Reconstruction(sparse)
    remaps: dict[int, tuple] = {}
    views = []
    for iid in sorted(rec.reg_image_ids(), key=lambda i: rec.images[i].name):
        im = rec.images[iid]
        cam = rec.cameras[im.camera_id]
        if im.camera_id not in remaps:
            remaps[im.camera_id] = _pinhole_remap(cam, long_side)
        mx, my, K, w, h = remaps[im.camera_id]

        bgr = cv2.imread(str(img_dir / im.name))
        if bgr.shape[1] != cam.width or bgr.shape[0] != cam.height:
            raise ValueError(f"{im.name}: image size differs from its COLMAP camera")
        rgb = cv2.remap(bgr, mx, my, cv2.INTER_AREA if long_side < cam.width else cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)[..., ::-1].copy()
        invalid = (mx < 0) | (my < 0) | (mx > cam.width - 1) | (my > cam.height - 1)
        mask = invalid
        mpath = mask_dir / f"{Path(im.name).stem}.png" if mask_dir else None
        if mpath and mpath.exists():
            m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            mask = mask | (cv2.remap(m, mx, my, cv2.INTER_NEAREST, borderValue=255) > 127)

        w2c = np.eye(4)
        w2c[:3, :4] = im.cam_from_world().matrix()
        views.append(View(im.name, rgb, mask, K, w2c))
    return views


def sparse_points(sparse: Path) -> tuple[np.ndarray, np.ndarray]:
    rec = pycolmap.Reconstruction(sparse)
    pts = rec.points3D.values()
    xyz = np.array([p.xyz for p in pts], dtype=np.float32)
    rgb = np.array([p.color for p in pts], dtype=np.float32) / 255.0
    return xyz, rgb
