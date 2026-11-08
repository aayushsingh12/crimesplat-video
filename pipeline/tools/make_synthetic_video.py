"""Render a synthetic walk-through of a textured room, with known camera poses.

Used to smoke-test the pipeline without real footage: the camera walks a loop at a fixed
height (default 1.5 m) through a 6 x 3 x 5 m room with two boxes of "furniture". Ground-truth
camera centres are written next to the video so S04/S05 output can be checked.

    uv run python tools/make_synthetic_video.py ../data/synthetic_room.mp4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECONDS = 960, 540, 15, 20
FOCAL = 600.0
ROOM = np.array([[-3.0, 0.0, -2.5], [3.0, 3.0, 2.5]])
BOXES = [np.array([[1.2, 0.0, 0.6], [2.4, 0.9, 1.8]]), np.array([[-2.6, 0.0, -2.2], [-1.6, 1.8, -1.6]])]
CAM_HEIGHT = 1.5
TEX = 2048  # covers 6.8 m at TEXELS_PER_M, so no face shows a repeated pattern
TEXELS_PER_M = 300


def make_texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((TEX, TEX, 3), np.float32)
    for octave in range(1, 7):
        n = 2 ** (octave + 1)
        layer = cv2.resize(rng.random((n, n, 3)).astype(np.float32), (TEX, TEX), interpolation=cv2.INTER_CUBIC)
        img += layer / octave
    img = (img - img.min()) / (img.max() - img.min())
    base = rng.random(3) * 0.6 + 0.2
    img = 0.5 * img + 0.5 * base
    img = (img * 255).astype(np.uint8)
    for _ in range(400):  # blobs and strokes give SIFT plenty of distinct features
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        p = tuple(int(x) for x in rng.integers(0, TEX, 2))
        if rng.random() < 0.5:
            cv2.circle(img, p, int(rng.integers(4, 30)), c, -1)
        else:
            q = tuple(int(x) for x in rng.integers(0, TEX, 2))
            cv2.line(img, p, q, c, int(rng.integers(1, 5)))
    return img


def intersect_aabb(o, d, box, inside: bool):
    """Ray/box hit distance and hit-face axis for N rays. inside=True for the room shell."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (box[0] - o) / d
        t2 = (box[1] - o) / d
    tmin, tmax = np.minimum(t1, t2), np.maximum(t1, t2)
    if inside:
        t = tmax.min(1)
        axis = tmax.argmin(1)
    else:
        tn, tf = tmin.max(1), tmax.min(1)
        hit = (tn <= tf) & (tn > 1e-4)
        t = np.where(hit, tn, np.inf)
        axis = tmin.argmax(1)
    return t, axis


def render(c2w: np.ndarray, textures: list[np.ndarray]) -> np.ndarray:
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dirs_cam = np.stack([(u - W / 2) / FOCAL, (v - H / 2) / FOCAL, np.ones_like(u)], -1).reshape(-1, 3)
    d = dirs_cam @ c2w[:3, :3].T
    o = np.broadcast_to(c2w[:3, 3], d.shape)

    # Every face gets its own texture id (axis and side), so no two surfaces look identical.
    rows = np.arange(len(d))
    t, axis = intersect_aabb(o, d, ROOM, inside=True)
    surf = axis * 2 + (d[rows, axis] > 0)
    for bi, box in enumerate(BOXES):
        tb, ab = intersect_aabb(o, d, box, inside=False)
        closer = tb < t
        t = np.where(closer, tb, t)
        surf = np.where(closer, 6 + 6 * bi + ab * 2 + (d[rows, ab] < 0), surf)

    p = o + d * t[:, None]
    out = np.zeros((len(p), 3), np.uint8)
    for s in np.unique(surf):
        sel = surf == s
        ax = (s // 2) % 3 if s < 6 else ((s - 6) % 6) // 2
        a, b = [i for i in range(3) if i != ax]
        tex = textures[int(s)]
        tu = (p[sel, a] * TEXELS_PER_M).astype(int) % TEX
        tv = (p[sel, b] * TEXELS_PER_M).astype(int) % TEX
        out[sel] = tex[tv, tu]
    shade = np.clip(1.2 - 0.06 * t, 0.5, 1.0)[:, None]  # mild distance falloff
    return (out * shade).astype(np.uint8).reshape(H, W, 3)


def look_at(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenCV camera (x right, y down, z forward) in a Y-up world."""
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], 1)
    c2w[:3, 3] = pos
    return c2w


def main(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    textures = [make_texture(i) for i in range(6 + 6 * len(BOXES))]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    centres = []
    n = FPS * SECONDS
    for i in range(n):
        s = i / n * 2 * np.pi
        pos = np.array([1.6 * np.cos(s), CAM_HEIGHT + 0.03 * np.sin(8 * s), 1.2 * np.sin(s)])
        look = s + np.pi / 2 + 0.6 * np.sin(3 * s)  # walking direction plus looking around
        target = pos + np.array([np.cos(look), -0.15, np.sin(look)])
        c2w = look_at(pos, target)
        writer.write(render(c2w, textures)[..., ::-1])
        centres.append({"frame_index": i, "t": i / FPS, "centre": pos.round(5).tolist()})
    writer.release()
    gt = {"camera_height_m": CAM_HEIGHT, "focal_px": FOCAL, "size": [W, H], "frames": centres}
    out_path.with_suffix(".gt.json").write_text(json.dumps(gt))
    print(f"wrote {out_path} ({n} frames) and ground truth")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "../data/synthetic_room.mp4"))
