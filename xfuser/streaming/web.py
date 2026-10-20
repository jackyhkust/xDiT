"""Browser playback and control for the live stream.

Besides the MPEG-TS for VLC, the streamer encodes the same playout as a
fragmented MP4 (H.264 + AAC, one fragment per GOP) that browsers play with
Media Source Extensions. ``Mp4FanOut`` splits the muxer's bytes into the init
segment (``ftyp`` + ``moov``) and fragments (``moof`` + ``mdat``); every
fragment starts on a keyframe, so a viewer can join at any of them.

The page at ``/`` (``static/index.html``) plays ``/stream.mp4``, keeps itself
close to live, and sends prompts to the same queue as the console and the demo
loop, so they are enhanced and cut in exactly like typed ones.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from xfuser.logger import init_logger
from xfuser.streaming.prompt_cli import clean_prompt
from xfuser.streaming.prompt_queue import append_prompt, read_new_prompts, toggle_overlay

logger = init_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"
MAX_PROMPT_BYTES = 16 * 1024
RECENT_PROMPTS = 8
MP4_MOVFLAGS = "frag_keyframe+empty_moov+default_base_moof"

_INIT_BOXES = (b"ftyp", b"moov")


def codecs_from_init(init: bytes) -> str:
    """The MSE codecs string (e.g. ``avc1.64001f,mp4a.40.2``) for an init segment."""
    video = "avc1.640028"
    index = init.find(b"avcC")
    # avcC: configurationVersion (1), then profile, profile compatibility, level.
    if index >= 0 and len(init) >= index + 8 and init[index + 4] == 1:
        video = f"avc1.{init[index + 5]:02x}{init[index + 6]:02x}{init[index + 7]:02x}"
    return f"{video},mp4a.40.2" if b"mp4a" in init else video


class Mp4FanOut:
    """File-like sink for a fragmented-MP4 muxer that hands whole fragments to every viewer.

    A viewer gets the init segment, then each fragment published after it
    joined. One that falls ``max_queued_fragments`` behind is dropped (its page
    reconnects at the live edge) rather than allowed to block the encoder.
    """

    def __init__(self, max_queued_fragments: int = 10) -> None:
        self.max_queued_fragments = max_queued_fragments
        self._buffer = bytearray()
        self._init = bytearray()
        self._fragment = bytearray()
        self._init_ready = threading.Event()
        self._clients: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self.fragments = 0

    def write(self, data) -> int:
        self._buffer += data
        while len(self._buffer) >= 8:
            size = int.from_bytes(self._buffer[:4], "big")
            header = 8
            if size == 1:
                if len(self._buffer) < 16:
                    break
                size, header = int.from_bytes(self._buffer[8:16], "big"), 16
            if size < header:
                raise ValueError(f"Malformed MP4 box of size {size}.")
            if len(self._buffer) < size:
                break
            kind = bytes(self._buffer[4:8])
            box = bytes(self._buffer[:size])
            del self._buffer[:size]
            self._on_box(kind, box)
        return len(data)

    def _on_box(self, kind: bytes, box: bytes) -> None:
        if kind in _INIT_BOXES and not self._init_ready.is_set():
            self._init += box
            if kind == b"moov":
                self._init_ready.set()
        elif kind == b"moof":
            self._fragment = bytearray(box)
        elif kind == b"mdat" and self._fragment:
            self._fragment += box
            self._publish(bytes(self._fragment))
            self._fragment = bytearray()

    def _publish(self, fragment: bytes) -> None:
        with self._lock:
            self.fragments += 1
            for client in list(self._clients):
                try:
                    client.put_nowait(fragment)
                except queue.Full:
                    logger.warning("Dropping a web viewer that fell behind the live stream.")
                    self._drop(client)

    def _drop(self, client: queue.Queue) -> None:
        self._clients.discard(client)
        with client.mutex:
            client.queue.clear()
        client.put_nowait(None)

    def add_client(self, timeout: float = 30.0) -> tuple[queue.Queue, bytes] | None:
        """Register a viewer and return its fragment queue and the init segment.

        Waits for the muxer to write the init segment; None if it has not after ``timeout``.
        """
        if not self._init_ready.wait(timeout):
            return None
        client: queue.Queue = queue.Queue(maxsize=self.max_queued_fragments)
        with self._lock:
            self._clients.add(client)
        return client, bytes(self._init)

    def remove_client(self, client: queue.Queue) -> None:
        with self._lock:
            self._clients.discard(client)

    @property
    def num_clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def close(self) -> None:
        """End every viewer after the fragments already queued for it."""
        with self._lock:
            for client in list(self._clients):
                self._clients.discard(client)
                try:
                    client.put_nowait(None)
                except queue.Full:
                    self._drop(client)


def status(streamer) -> dict:
    """What the page shows: playback health, the prompt pipeline and recent prompts."""
    summary = streamer.summary()
    latencies = summary["prompt_latencies_s"]
    recent = summary["recent_chunks"]
    prompts = read_new_prompts(streamer.stream_dir, 0)[-RECENT_PROMPTS:]
    return {
        "started": summary["started"],
        "buffered_s": summary["buffered_s"],
        "stall_frames": summary["stall_frames"],
        "content_frames": summary["content_frames"],
        "chunks_played": summary["chunks_played"],
        "last_gen_time_s": recent[-1].get("gen_time_s") if recent else None,
        "overlay": summary["overlay"],
        "viewers": {"ts": summary["clients"], "web": summary["web_clients"]},
        "on_screen_prompt_id": summary["played_prompt_id"],
        "prompt_state": summary["prompt_state"],
        "prompts": [
            {
                "id": prompt.id,
                "text": prompt.text,
                "source": prompt.source,
                "ts": prompt.ts,
                "latency_s": latencies.get(prompt.id),
            }
            for prompt in reversed(prompts)
        ],
        "video": {"width": streamer.width, "height": streamer.height, "fps": streamer.fps},
        "server_time": time.time(),
    }


def submit(stream_dir: Path, text: str) -> tuple[int, dict]:
    """Handle the page's prompt box: queue a prompt, or run ``/overlay``. Returns (HTTP status, body)."""
    text = clean_prompt(text)
    if not text:
        return 400, {"error": "The prompt is empty."}
    if text == "/overlay":
        return 200, {"overlay": toggle_overlay(stream_dir)}
    if text.startswith("/"):
        return 400, {"error": f"Unknown command {text.split()[0]!r}; the page knows /overlay."}
    prompt = append_prompt(stream_dir, text, source="web")
    logger.info(f"Web viewer queued prompt #{prompt.id}: {prompt.text!r}")
    return 200, {"id": prompt.id, "text": prompt.text}
