"""Chunk handoff between the generator and the streamer through a shared directory.

Each chunk is one ``.npz`` file written atomically (tmp file, then rename) so the
reader never sees a partial chunk. The streamer reports how many chunks it has
played in a small text file, which the generator reads for backpressure.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PLAYED_FILE = "played"
DONE_FILE = "done"


@dataclass
class Chunk:
    index: int
    frames: np.ndarray  # uint8 [F, H, W, 3]
    audio: np.ndarray | None  # float32 [2, S] in [-1, 1]
    sample_rate: int
    fps: int
    meta: dict = field(default_factory=dict)


def chunk_path(stream_dir: Path, index: int) -> Path:
    return Path(stream_dir) / f"chunk_{index:05d}.npz"


def write_chunk(stream_dir: Path, chunk: Chunk) -> Path:
    stream_dir = Path(stream_dir)
    stream_dir.mkdir(parents=True, exist_ok=True)
    final_path = chunk_path(stream_dir, chunk.index)
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    arrays = {
        "frames": chunk.frames,
        "sample_rate": np.int64(chunk.sample_rate),
        "fps": np.int64(chunk.fps),
        "meta": np.frombuffer(json.dumps(chunk.meta).encode(), dtype=np.uint8),
    }
    if chunk.audio is not None:
        arrays["audio"] = chunk.audio.astype(np.float32, copy=False)
    with open(tmp_path, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp_path, final_path)
    return final_path


def read_chunk(stream_dir: Path, index: int, delete: bool = False) -> Chunk | None:
    """Return chunk ``index`` if it has been fully written, else None."""
    path = chunk_path(stream_dir, index)
    if not path.exists():
        return None
    with np.load(path) as data:
        chunk = Chunk(
            index=index,
            frames=data["frames"],
            audio=data["audio"] if "audio" in data.files else None,
            sample_rate=int(data["sample_rate"]),
            fps=int(data["fps"]),
            meta=json.loads(data["meta"].tobytes().decode()),
        )
    if delete:
        path.unlink()
    return chunk


def write_played(stream_dir: Path, count: int) -> None:
    stream_dir = Path(stream_dir)
    tmp_path = stream_dir / f"{PLAYED_FILE}.tmp"
    tmp_path.write_text(str(count))
    os.replace(tmp_path, stream_dir / PLAYED_FILE)


def read_played(stream_dir: Path) -> int:
    path = Path(stream_dir) / PLAYED_FILE
    try:
        return int(path.read_text())
    except (FileNotFoundError, ValueError):
        return 0


def mark_done(stream_dir: Path, num_chunks: int) -> None:
    """Tell the streamer no chunk past ``num_chunks - 1`` is coming."""
    (Path(stream_dir) / DONE_FILE).write_text(str(num_chunks))


def read_done(stream_dir: Path) -> int | None:
    try:
        return int((Path(stream_dir) / DONE_FILE).read_text())
    except (FileNotFoundError, ValueError):
        return None


def reset_stream_dir(stream_dir: Path) -> None:
    """Remove chunks and markers from a previous run."""
    stream_dir = Path(stream_dir)
    stream_dir.mkdir(parents=True, exist_ok=True)
    for path in stream_dir.iterdir():
        if path.name.startswith("chunk_") or path.name in (PLAYED_FILE, DONE_FILE):
            path.unlink()
