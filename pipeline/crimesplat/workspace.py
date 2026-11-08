"""Per-case workspace: stage output dirs, cache keys and the provenance log."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Workspace:
    root: Path  # workspace/<case_id>

    def stage_dir(self, stage: str) -> Path:
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _marker(self, stage: str) -> Path:
        return self.root / stage / "_SUCCESS.json"

    def cache_key(self, stage: str) -> str | None:
        m = self._marker(stage)
        return json.loads(m.read_text())["cache_key"] if m.exists() else None

    def is_fresh(self, stage: str, key: str) -> bool:
        return self.cache_key(stage) == key

    def invalidate(self, stage: str) -> None:
        self._marker(stage).unlink(missing_ok=True)

    def mark_done(self, stage: str, key: str, report: dict, started: float) -> None:
        payload = {"cache_key": key, "finished_at": time.time(), "duration_s": round(time.time() - started, 1)}
        self._marker(stage).write_text(json.dumps(payload, indent=2))
        (self.root / stage / "report.json").write_text(json.dumps(report, indent=2, default=str))
        self.log_provenance(stage, key, payload["duration_s"], report.get("status", "ok"))

    def log_provenance(self, stage: str, key: str, duration_s: float, status: str) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage,
            "cache_key": key,
            "pipeline_version": __version__,
            "host": platform.node(),
            "gpu": _gpu_name(),
            "duration_s": duration_s,
            "status": status,
        }
        with open(self.root / "provenance.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return f"{props.name} ({props.total_memory // 2**20} MiB)"
    except Exception:
        pass
    return None
