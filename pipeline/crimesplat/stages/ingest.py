"""S01 Ingest: fetch the source video, hash it, and probe its metadata.

The original bytes are never modified; everything downstream reads `source.<ext>`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..config import CaseConfig
from ..workspace import sha256_file


def _is_url(uri: str) -> bool:
    return uri.startswith(("http://", "https://"))


def _download(url: str, out_dir: Path) -> tuple[Path, dict]:
    from yt_dlp import YoutubeDL

    opts = {
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    path = next(out_dir.glob("source.*"))
    meta = {k: info.get(k) for k in ("id", "title", "uploader", "upload_date", "webpage_url", "duration")}
    return path, meta


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in video["avg_frame_rate"].split("/"))
    rotation = 0
    for sd in video.get("side_data_list", []):
        rotation = int(sd.get("rotation", rotation))
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": num / den if den else None,
        "variable_frame_rate": video["avg_frame_rate"] != video.get("r_frame_rate"),
        "duration_s": float(data["format"]["duration"]),
        "codec": video["codec_name"],
        "rotation": rotation,
        "has_audio": any(s["codec_type"] == "audio" for s in data["streams"]),
    }


def run(cfg: CaseConfig, out_dir: Path) -> dict:
    if len(cfg.sources) != 1:
        raise NotImplementedError("multi-source cases are not supported yet; pass exactly one source")
    src = cfg.sources[0]
    for old in out_dir.glob("source.*"):
        old.unlink()

    if _is_url(src.uri):
        path, origin = _download(src.uri, out_dir)
    else:
        local = Path(src.uri).expanduser().resolve()
        if not local.exists():
            raise FileNotFoundError(local)
        path = out_dir / f"source{local.suffix.lower()}"
        shutil.copy2(local, path)
        origin = {"local_path": str(local)}

    meta = {
        "camera_id": src.camera_id,
        "uri": src.uri,
        "origin": origin,
        "file": path.name,
        "sha256": sha256_file(path),
        "probe": probe(path),
    }
    (out_dir / "source.json").write_text(json.dumps(meta, indent=2))
    return {"status": "ok", **meta}
