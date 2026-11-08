"""S02 Frame selection: pick sharp keyframes spread evenly over the video.

Candidates are decoded at `candidate_fps`. The timeline is split into `target_keyframes`
bins and the sharpest candidate in each bin is kept, unless even that one is blurrier than
the clip-wide `min_sharpness_pct` percentile.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..config import CaseConfig


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian on a fixed-size image, so scores compare across resolutions."""
    small = cv2.resize(gray, (640, int(640 * gray.shape[0] / gray.shape[1])), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def _resize_long(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def _detect_shots(video: Path) -> list[float]:
    """Return start times (s) of hard cuts. Edited footage needs per-shot handling later."""
    try:
        from scenedetect import ContentDetector, detect

        return [s.get_seconds() for s, _ in detect(str(video), ContentDetector(threshold=27.0))]
    except Exception:
        return []


def run(cfg: CaseConfig, video: Path, out_dir: Path) -> dict:
    fc = cfg.frames
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for old in img_dir.glob("*.jpg"):
        old.unlink()

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = int(fc.start_s * fps)
    end = min(n_total, int(fc.end_s * fps)) if fc.end_s else n_total
    step = max(1, round(fps / fc.candidate_fps))

    # Pass 1: score candidates (grayscale only, cheap).
    cands: list[tuple[int, float]] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    while idx < end:
        ok = cap.grab()
        if not ok:
            break
        if (idx - start) % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                cands.append((idx, sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
        idx += 1
    if not cands:
        raise RuntimeError(f"no frames decoded from {video}")

    scores = np.array([s for _, s in cands])
    floor = np.percentile(scores, fc.min_sharpness_pct)
    bins = np.array_split(np.arange(len(cands)), min(fc.target_keyframes, len(cands)))
    chosen = []
    for b in bins:
        best = b[np.argmax(scores[b])]
        if scores[best] >= floor:
            chosen.append(cands[best][0])

    # Pass 2: write the chosen frames at full quality.
    frames = []
    for fidx in chosen:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        name = f"f{fidx:06d}.jpg"
        cv2.imwrite(str(img_dir / name), _resize_long(frame, fc.max_long_side), [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append({"name": name, "frame_index": fidx, "t": fidx / fps,
                       "sharpness": dict(cands)[fidx]})
    cap.release()

    shots = _detect_shots(video)
    index = {"fps": fps, "n_video_frames": n_total, "candidates": len(cands),
             "sharpness_floor": floor, "shot_starts_s": shots, "keyframes": frames}
    (out_dir / "frames.json").write_text(json.dumps(index, indent=2))

    warnings = []
    if len(shots) > 1:
        warnings.append(f"{len(shots)} shots detected; edited footage may not register as one scene")
    if len(frames) < 30:
        warnings.append(f"only {len(frames)} keyframes; reconstruction will likely fail")
    return {"status": "degraded" if warnings else "ok", "keyframes": len(frames),
            "candidates": len(cands), "shots": len(shots), "warnings": warnings}
