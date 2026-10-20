"""Runtime prompt handoff between the interactive CLI and the stream generator.

The CLI appends user prompts to ``prompts.jsonl`` in the stream directory. The
generator's writer rank reads new lines before each chunk and keeps only the
newest queued prompt. The generator also publishes ``prompt_state.json`` so the
streamer's overlay can show the latest user input and where it is in its life
cycle (queued -> enhancing -> rendering -> live).

Both files are written atomically (tmp file, then ``os.replace``) so a reader
never sees a half-written record. ``prompts.jsonl`` is append-only; the writer
rank remembers the last id it has consumed instead of truncating the file, so
the CLI and the generator never contend on it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROMPTS_FILE = "prompts.jsonl"
PROMPT_STATE_FILE = "prompt_state.json"
# "1" or "0": whether the streamer draws its overlay; flipped by the CLI's /overlay.
OVERLAY_FILE = "overlay"

# Life-cycle states published in prompt_state.json.
STATE_QUEUED = "queued"
STATE_ENHANCING = "enhancing"
STATE_RENDERING = "rendering"
STATE_LIVE = "live"


@dataclass
class UserPrompt:
    id: int
    text: str
    ts: float


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def append_prompt(stream_dir: Path, text: str) -> UserPrompt:
    """Append one user prompt; its id is the 1-based line number in the file."""
    stream_dir = Path(stream_dir)
    stream_dir.mkdir(parents=True, exist_ok=True)
    path = stream_dir / PROMPTS_FILE
    next_id = _count_lines(path) + 1
    prompt = UserPrompt(id=next_id, text=text.strip(), ts=time.time())
    with open(path, "a") as handle:
        handle.write(json.dumps(asdict(prompt)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return prompt


def _count_lines(path: Path) -> int:
    try:
        with open(path) as handle:
            return sum(1 for line in handle if line.strip())
    except FileNotFoundError:
        return 0


def read_new_prompts(stream_dir: Path, after_id: int) -> list[UserPrompt]:
    """All prompts whose id is greater than ``after_id``, oldest first."""
    path = Path(stream_dir) / PROMPTS_FILE
    prompts: list[UserPrompt] = []
    try:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("id", 0) > after_id:
                    prompts.append(UserPrompt(id=record["id"], text=record["text"], ts=record["ts"]))
    except FileNotFoundError:
        return []
    return prompts


def write_prompt_state(stream_dir: Path, state: dict) -> None:
    _atomic_write_text(Path(stream_dir) / PROMPT_STATE_FILE, json.dumps(state))


def read_prompt_state(stream_dir: Path) -> dict | None:
    path = Path(stream_dir) / PROMPT_STATE_FILE
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def latest_prompt(stream_dir: Path) -> UserPrompt | None:
    """The most recently typed prompt, whether or not the generator has picked it up yet."""
    prompts = read_new_prompts(stream_dir, 0)
    return prompts[-1] if prompts else None


def write_overlay(stream_dir: Path, enabled: bool) -> None:
    stream_dir = Path(stream_dir)
    stream_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(stream_dir / OVERLAY_FILE, "1" if enabled else "0")


def read_overlay(stream_dir: Path) -> bool | None:
    """Whether the streamer should draw the overlay; None if no streamer has published a state."""
    try:
        return (Path(stream_dir) / OVERLAY_FILE).read_text().strip() == "1"
    except FileNotFoundError:
        return None


def toggle_overlay(stream_dir: Path) -> bool:
    """Flip the overlay switch and return the new state."""
    enabled = not read_overlay(stream_dir)
    write_overlay(stream_dir, enabled)
    return enabled


def clear_prompt_files(stream_dir: Path) -> None:
    """Remove the prompt queue, state and overlay markers from a previous run."""
    stream_dir = Path(stream_dir)
    for name in (PROMPTS_FILE, PROMPT_STATE_FILE, OVERLAY_FILE):
        path = stream_dir / name
        if path.exists():
            path.unlink()
