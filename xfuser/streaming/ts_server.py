"""Real-time MPEG-TS streamer for generated chunks.

Reads ``chunk_*.npz`` files from a stream directory in order, stitches them into
one continuous timeline (``ChunkStitcher``), and plays that timeline out at a
fixed frame rate. It encodes once (H.264 + AAC in MPEG-TS) and fans the bytes
out to every HTTP client, and encodes a second copy as fragmented MP4 for the
web page at ``/`` (see ``xfuser.streaming.web``)::

    python -m xfuser.streaming.ts_server --stream_dir /dev/shm/h3_stream --port 20123
    vlc http://<host>:20123/stream.ts      # or open http://<host>:20123/ in Chrome

The playout clock never waits for the generator. Until the first chunks arrive
it shows the anchor image. When the buffer runs dry it repeats the last frame
with silence and counts each one in ``stall_frames``; a sustained real-time
stream keeps that at 0. ``GET /stats`` reports buffer depth, stalls, generation
times and drift metrics.

When the generator writes a cut marker for a new prompt, the buffered frames
after the cut frame are dropped and the next chunk continues from it. The
streamer publishes its exact playback position (content frames played) so the
generator can pick that cut frame.
"""

from __future__ import annotations

import argparse
import collections
import json
import queue
import signal
import threading
import time
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from xfuser.logger import init_logger
from xfuser.streaming.chunks import (
    read_chunk,
    read_cut,
    read_done,
    reset_stream_dir,
    write_played,
    write_position,
)
from xfuser.streaming.drift import chunk_metrics
from xfuser.streaming.overlay import Overlay
from xfuser.streaming.prompt_queue import (
    latest_prompt,
    read_new_prompts,
    read_overlay,
    read_prompt_state,
    toggle_overlay,
    write_overlay,
)
from xfuser.streaming.resolution import cover_resize, upright
from xfuser.streaming.stitch import ChunkStitcher
from xfuser.streaming.web import (
    INDEX_HTML,
    MAX_PROMPT_BYTES,
    MP4_MOVFLAGS,
    Mp4FanOut,
    codecs_from_init,
)
from xfuser.streaming.web import status as web_status
from xfuser.streaming.web import submit as submit_prompt

logger = init_logger(__name__)

TS_PACKET_SIZE = 188


class FanOut:
    """File-like sink for the muxer that copies the byte stream to every client.

    Bytes are batched per frame (``flush_frame``) so each client queue entry is
    one frame's worth of TS packets. A client that falls ``max_queued_frames``
    behind is dropped rather than allowed to block the encoder.
    """

    def __init__(self, max_queued_frames: int, record_path: Path | None = None) -> None:
        self.max_queued_frames = max_queued_frames
        self._pending = bytearray()
        self._clients: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self._bytes_written = 0
        self._record = open(record_path, "wb") if record_path else None

    def write(self, data) -> int:
        self._pending += data
        return len(data)

    def flush_frame(self) -> None:
        if not self._pending:
            return
        data = bytes(self._pending)
        self._pending.clear()
        if self._record:
            self._record.write(data)
        with self._lock:
            for client in list(self._clients):
                try:
                    client.put_nowait(data)
                except queue.Full:
                    logger.warning("Dropping a client that fell behind the live stream.")
                    self._clients.discard(client)
                    with client.mutex:
                        client.queue.clear()
                    client.put_nowait(None)
            self._bytes_written += len(data)

    def add_client(self) -> tuple[queue.Queue, int]:
        """Register a client; also return how many bytes to skip to reach a TS packet boundary."""
        client: queue.Queue = queue.Queue(maxsize=self.max_queued_frames)
        with self._lock:
            self._clients.add(client)
            skip = -self._bytes_written % TS_PACKET_SIZE
        return client, skip

    def remove_client(self, client: queue.Queue) -> None:
        with self._lock:
            self._clients.discard(client)

    @property
    def num_clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def close(self) -> None:
        with self._lock:
            for client in self._clients:
                try:
                    client.put_nowait(None)
                except queue.Full:
                    pass
            self._clients.clear()
        if self._record:
            self._record.close()


class AudioBuffer:
    def __init__(self, channels: int) -> None:
        self.channels = channels
        self._pieces: collections.deque[np.ndarray] = collections.deque()
        self.available = 0

    def append(self, samples: np.ndarray) -> None:
        if len(samples):
            self._pieces.append(samples)
            self.available += len(samples)

    def read(self, count: int) -> np.ndarray:
        """Exactly ``count`` samples, zero-padded if the buffer runs short."""
        out = np.zeros((count, self.channels), dtype=np.float32)
        filled = 0
        while filled < count and self._pieces:
            piece = self._pieces[0]
            take = min(count - filled, len(piece))
            out[filled : filled + take] = piece[:take]
            filled += take
            if take == len(piece):
                self._pieces.popleft()
            else:
                self._pieces[0] = piece[take:]
        self.available -= filled
        return out

    def truncate(self, keep: int, fade: int = 0) -> None:
        """Keep only the first ``keep`` samples, fading out the last ``fade`` of them."""
        kept: collections.deque[np.ndarray] = collections.deque()
        total = 0
        for piece in self._pieces:
            if total >= keep:
                break
            piece = piece[: keep - total]
            kept.append(piece)
            total += len(piece)
        self._pieces = kept
        self.available = total
        fade = min(fade, total)
        if fade and kept:
            tail = np.concatenate(list(kept), axis=0)
            tail[total - fade :] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
            self._pieces = collections.deque([tail])


def load_still(path: str | None, width: int, height: int) -> np.ndarray:
    if not path:
        return np.zeros((height, width, 3), dtype=np.uint8)
    from PIL import Image

    return np.asarray(cover_resize(upright(Image.open(path)).convert("RGB"), (width, height)))


class Streamer:
    def __init__(
        self,
        stream_dir: Path,
        width: int,
        height: int,
        fps: int = 24,
        sample_rate: int = 32000,
        channels: int = 2,
        anchor_image: str | None = None,
        preroll_chunks: int = 1,
        delete_chunks: bool = True,
        metrics_path: Path | None = None,
        record_path: Path | None = None,
        crf: int = 23,
        max_kbps: int = 3000,
        preset: str = "veryfast",
        gop: int | None = None,
        audio_bitrate: int = 192_000,
        max_client_lag_s: float = 10.0,
        overlay: bool = False,
        web: bool = True,
    ) -> None:
        self.stream_dir = Path(stream_dir)
        self.width, self.height = width, height
        self.fps = fps
        self.sample_rate = sample_rate
        self.channels = channels
        self.preroll_chunks = preroll_chunks
        self.delete_chunks = delete_chunks
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.crf, self.preset, self.gop = crf, preset, gop or fps
        self.max_kbps = max_kbps
        self.audio_bitrate = audio_bitrate

        self.anchor = load_still(anchor_image, width, height)
        self.sink = FanOut(max_queued_frames=int(max_client_lag_s * fps), record_path=record_path)
        # A second encode of the same playout as fragmented MP4 for the web page.
        self.web = Mp4FanOut(max_queued_fragments=max(2, round(max_client_lag_s * fps / self.gop))) if web else None
        self.stitcher = ChunkStitcher(fps, sample_rate, channels)

        self.overlay = Overlay(fps)
        self.overlay_enabled = overlay
        write_overlay(self.stream_dir, overlay)
        self._latest_input = ""
        self._prompt_state: dict | None = None
        self._controls_ts = 0.0
        self._chunk_prompt_id: dict[int, int] = {}
        self.played_prompt_id = 0

        self._lock = threading.Lock()
        # (frame, chunk index if this is the chunk's last frame, chunk index if its first)
        self._frames: collections.deque[tuple[np.ndarray, int | None, int | None]] = collections.deque()
        self._audio = AudioBuffer(channels)
        self._stop = threading.Event()
        self.finished = threading.Event()

        self.started = False
        self.frames_emitted = 0
        self.content_frames = 0
        self.stall_frames = 0
        self.chunks_loaded = 0
        self.chunks_played = 0
        self.max_lag_s = 0.0
        self.end_of_stream = False
        self.recent_chunks: collections.deque[dict] = collections.deque(maxlen=16)
        self.cut_epoch = 0
        self.cuts: collections.deque[dict] = collections.deque(maxlen=16)
        self.prompt_latencies: dict[int, float] = {}

    def _apply_cut(self, cut: dict) -> np.ndarray | None:
        """Drop buffered frames after ``cut['keep_through']``; return the new last buffered frame."""
        with self._lock:
            keep = max(0, cut["keep_through"] + 1 - self.content_frames)
            dropped = max(0, len(self._frames) - keep)
            late = cut["keep_through"] + 1 - self.content_frames < 0
            while len(self._frames) > keep:
                self._frames.pop()
            if self._frames:
                frame, _, started = self._frames[-1]
                self._frames[-1] = (frame, cut["after_chunk"], started)
            else:
                self.chunks_played = cut["after_chunk"] + 1
                write_played(self.stream_dir, self.chunks_played)
            boundary = lambda j: round(j * self.sample_rate / self.fps)  # noqa: E731
            keep_samples = boundary(self.content_frames + keep) - boundary(self.content_frames)
            fade = boundary(self.content_frames + keep) - boundary(self.content_frames + max(0, keep - 1))
            self._audio.truncate(keep_samples, fade=fade)
            self.stitcher.rewind(self.content_frames + keep)
            self.cut_epoch = cut["epoch"]
            record = {
                "epoch": cut["epoch"],
                "prompt_id": cut.get("prompt_id"),
                "keep_through": cut["keep_through"],
                "played_at_cut": self.content_frames,
                "dropped_s": round(dropped / self.fps, 3),
                "late_frames": max(0, self.content_frames - cut["keep_through"] - 1),
            }
            self.cuts.append(record)
            last = self._frames[-1][0] if self._frames else None
        message = (
            f"Cut-in for prompt #{record['prompt_id']}: keeping frames through {cut['keep_through']} "
            f"({keep / self.fps:.2f}s left to play), dropped {record['dropped_s']:.2f}s of buffered video."
        )
        if late:
            logger.warning(f"{message} The cut arrived {record['late_frames']} frames late.")
        else:
            logger.info(message)
        return last

    # Loader: chunk files -> stitched frame/audio buffers
    def _load_loop(self) -> None:
        index = 0
        previous_last = None
        while not self._stop.is_set():
            cut = read_cut(self.stream_dir)
            if cut is not None and cut["epoch"] > self.cut_epoch and index > cut["after_chunk"]:
                last = self._apply_cut(cut)
                if last is not None:
                    previous_last = last
            chunk = read_chunk(self.stream_dir, index, delete=self.delete_chunks)
            if chunk is None:
                done = read_done(self.stream_dir)
                if done is not None and index >= done:
                    with self._lock:
                        self.end_of_stream = True
                    return
                time.sleep(0.02)
                continue
            if chunk.fps != self.fps or chunk.sample_rate != self.sample_rate:
                raise ValueError(
                    f"Chunk {index} is {chunk.fps} fps / {chunk.sample_rate} Hz; "
                    f"the stream is {self.fps} fps / {self.sample_rate} Hz."
                )
            if chunk.frames.shape[1:3] != (self.height, self.width):
                raise ValueError(
                    f"Chunk {index} frames are {chunk.frames.shape[2]}x{chunk.frames.shape[1]}; "
                    f"the stream is {self.width}x{self.height}."
                )
            record = {
                "index": index,
                **chunk.meta,
                **chunk_metrics(chunk.frames, self.anchor, previous_last),
            }
            previous_last = chunk.frames[-1]
            self._chunk_prompt_id[index] = int(chunk.meta.get("prompt_id", 0))
            with self._lock:
                frames, audio = self.stitcher.push(chunk.frames, chunk.audio)
                for position, frame in enumerate(frames):
                    self._frames.append(
                        (
                            frame,
                            index if position == len(frames) - 1 else None,
                            index if position == 0 else None,
                        )
                    )
                self._audio.append(audio)
                self.chunks_loaded += 1
                record["buffered_s"] = round(len(self._frames) / self.fps, 3)
                self.recent_chunks.append(record)
            logger.info(
                f"chunk {index}: gen {chunk.meta.get('gen_time_s', float('nan')):.2f}s, "
                f"buffered {record['buffered_s']:.2f}s, seam PSNR {record.get('seam_psnr', float('nan')):.1f} dB"
            )
            if self.metrics_path:
                with open(self.metrics_path, "a") as handle:
                    handle.write(json.dumps(record) + "\n")
            index += 1

    # Playout clock: one frame every 1/fps seconds, whether or not content is ready
    def _next_frame(self, last_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        samples_per_frame = round((self.frames_emitted + 1) * self.sample_rate / self.fps) - round(
            self.frames_emitted * self.sample_rate / self.fps
        )
        with self._lock:
            if not self.started and self.chunks_loaded > 0:
                self.started = self.chunks_loaded >= self.preroll_chunks or self.end_of_stream
            if self.started and self._frames:
                frame, finished_chunk, started_chunk = self._frames.popleft()
                if started_chunk is not None:
                    self._on_chunk_start(started_chunk)
                content_samples = round((self.content_frames + 1) * self.sample_rate / self.fps) - round(
                    self.content_frames * self.sample_rate / self.fps
                )
                if self._audio.available < content_samples:
                    # The next chunk is late; release the held-back seam audio now.
                    self._audio.append(self.stitcher.flush())
                audio = self._audio.read(content_samples)
                self.content_frames += 1
                write_position(self.stream_dir, self.content_frames)
                if finished_chunk is not None:
                    self.chunks_played = finished_chunk + 1
                    write_played(self.stream_dir, self.chunks_played)
                return frame, audio, True
            if self.started and not self.end_of_stream:
                self.stall_frames += 1
                if self.stall_frames == 1 or self.stall_frames % self.fps == 0:
                    logger.warning(f"Underrun: repeating the last frame ({self.stall_frames} stall frames so far).")
            elif self.end_of_stream and self.started:
                self.finished.set()
        still = last_frame if self.started else self.anchor
        return still, np.zeros((samples_per_frame, self.channels), dtype=np.float32), False

    def _on_chunk_start(self, index: int) -> None:
        """A chunk's first frame is going on screen; log when a viewer prompt first becomes visible."""
        prompt_id = self._chunk_prompt_id.pop(index, 0)
        if prompt_id <= self.played_prompt_id:
            return
        self.played_prompt_id = prompt_id
        typed = read_new_prompts(self.stream_dir, prompt_id - 1)
        if typed and typed[0].id == prompt_id:
            latency = time.time() - typed[0].ts
            self.prompt_latencies[prompt_id] = round(latency, 2)
            logger.info(f"Prompt #{prompt_id} on screen {latency:.1f}s after it was typed.")

    def _poll_controls(self) -> None:
        """Every 0.5 s, re-read the overlay switch, the latest typed prompt and the prompt state."""
        now = time.monotonic()
        if now - self._controls_ts < 0.5:
            return
        self._controls_ts = now
        enabled = read_overlay(self.stream_dir)
        if enabled is not None and enabled != self.overlay_enabled:
            self.overlay_enabled = enabled
            logger.info(f"Overlay {'on' if enabled else 'off'}.")
        latest = latest_prompt(self.stream_dir)
        self._latest_input = latest.text if latest else ""
        self._prompt_state = read_prompt_state(self.stream_dir)

    def _apply_overlay(self, frame: np.ndarray) -> np.ndarray:
        with self._lock:
            buffered = len(self._frames)
        return self.overlay.render(frame, buffered_frames=buffered, user_text=self._latest_input)

    def _open_output(self, sink, format: str):
        import av

        if format == "mp4":
            # One self-describing fragment per GOP, as Media Source Extensions expect.
            container = av.open(sink, mode="w", format="mp4", options={"movflags": MP4_MOVFLAGS})
        else:
            container = av.open(sink, mode="w", format=format)
        video = container.add_stream("libx264", rate=self.fps)
        video.width, video.height = self.width, self.height
        video.pix_fmt = "yuv420p"
        video.time_base = Fraction(1, self.fps)
        video.options = {
            "preset": self.preset,
            "tune": "zerolatency",
            "crf": str(self.crf),
            "g": str(self.gop),
            "keyint_min": str(self.gop),
            "sc_threshold": "0",
        }
        if format == "mpegts":
            # In-band SPS/PPS on every keyframe, so a TS client can join at any GOP.
            video.options["x264-params"] = "repeat-headers=1"
        if self.max_kbps:
            # Capped CRF: keeps quality-driven rate control but clips scene-change
            # spikes that would make a thin link (e.g. an SSH tunnel) fall behind.
            video.options |= {"maxrate": f"{self.max_kbps}k", "bufsize": f"{self.max_kbps}k"}
        audio = container.add_stream("aac", rate=self.sample_rate)
        audio.layout = "stereo" if self.channels == 2 else "mono"
        audio.bit_rate = self.audio_bitrate
        return container, video, audio

    def _play_loop(self) -> None:
        import av

        outputs = [self._open_output(self.sink, "mpegts")]
        if self.web is not None:
            outputs.append(self._open_output(self.web, "mp4"))
        fifo = av.AudioFifo()
        layout = "stereo" if self.channels == 2 else "mono"
        audio_pts = 0
        last_frame = self.anchor
        start = time.monotonic()
        try:
            while not self._stop.is_set():
                lag = time.monotonic() - (start + self.frames_emitted / self.fps)
                if lag < 0:
                    time.sleep(-lag)
                else:
                    self.max_lag_s = max(self.max_lag_s, lag)

                frame, samples, _ = self._next_frame(last_frame)
                last_frame = frame
                self._poll_controls()
                if self.overlay_enabled:
                    frame = self._apply_overlay(frame)

                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                video_frame.pts = self.frames_emitted
                for container, video_stream, _ in outputs:
                    for packet in video_stream.encode(video_frame):
                        container.mux(packet)

                audio_frame = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(samples).reshape(1, -1), format="flt", layout=layout
                )
                audio_frame.sample_rate = self.sample_rate
                fifo.write(audio_frame)
                while fifo.samples >= 1024:
                    block = fifo.read(1024)
                    block.pts = audio_pts
                    block.time_base = Fraction(1, self.sample_rate)
                    audio_pts += block.samples
                    for container, _, audio_stream in outputs:
                        for packet in audio_stream.encode(block):
                            container.mux(packet)

                self.frames_emitted += 1
                self.sink.flush_frame()
                if self.finished.is_set():
                    break
        finally:
            for container, video_stream, audio_stream in outputs:
                for packet in video_stream.encode():
                    container.mux(packet)
                for packet in audio_stream.encode():
                    container.mux(packet)
                container.close()
            self.sink.flush_frame()
            self.sink.close()
            if self.web is not None:
                self.web.close()
            self.finished.set()
            logger.info(f"Playout stopped: {self.summary()}")

    def summary(self) -> dict:
        with self._lock:
            return {
                "started": self.started,
                "end_of_stream": self.end_of_stream,
                "frames_emitted": self.frames_emitted,
                "content_frames": self.content_frames,
                "stall_frames": self.stall_frames,
                "chunks_loaded": self.chunks_loaded,
                "chunks_played": self.chunks_played,
                "played_prompt_id": self.played_prompt_id,
                "prompt_latencies_s": dict(self.prompt_latencies),
                "cuts": list(self.cuts),
                "prompt_state": self._prompt_state,
                "overlay": self.overlay_enabled,
                "buffered_s": round(len(self._frames) / self.fps, 3),
                "max_clock_lag_s": round(self.max_lag_s, 4),
                "clients": self.sink.num_clients,
                "web_clients": self.web.num_clients if self.web is not None else 0,
                "recent_chunks": list(self.recent_chunks),
            }

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._load_loop, name="chunk-loader", daemon=True),
            threading.Thread(target=self._play_loop, name="playout", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5)


def make_handler(streamer: Streamer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/stream.ts":
                self._stream()
            elif path == "/stream.mp4" and streamer.web is not None:
                self._stream_mp4()
            elif path in ("/", "/index.html") and streamer.web is not None:
                self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
            elif path == "/stats":
                self._send_json(200, streamer.summary())
            elif path == "/api/status":
                self._send_json(200, web_status(streamer))
            else:
                self.send_error(404)

        def do_POST(self):
            path = self.path.split("?")[0]
            if path == "/api/overlay":
                self._send_json(200, {"overlay": toggle_overlay(streamer.stream_dir)})
            elif path == "/api/prompt":
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_PROMPT_BYTES:
                    self._send_json(413, {"error": "The prompt is too long."})
                    return
                try:
                    text = str(json.loads(self.rfile.read(length) or b"{}").get("prompt", ""))
                except (ValueError, AttributeError):
                    self._send_json(400, {"error": 'Send JSON like {"prompt": "..."}.'})
                    return
                self._send_json(*submit_prompt(streamer.stream_dir, text))
            else:
                self.send_error(404)

        def _send(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, payload: dict):
            self._send(code, json.dumps(payload, indent=2).encode(), "application/json")

        def _stream_mp4(self):
            joined = streamer.web.add_client(timeout=30)
            if joined is None:
                self.send_error(503, "The stream has not started yet.")
                return
            client, init = joined
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("X-Codecs", codecs_from_init(init))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            logger.info(f"Web viewer {self.client_address[0]} connected.")
            try:
                self.wfile.write(init)
                while (fragment := client.get()) is not None:
                    self.wfile.write(fragment)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                streamer.web.remove_client(client)
                logger.info(f"Web viewer {self.client_address[0]} disconnected.")

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "video/mp2t")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            client, skip = streamer.sink.add_client()
            logger.info(f"Client {self.client_address[0]} connected.")
            try:
                while True:
                    data = client.get()
                    if data is None:
                        break
                    if skip:
                        data, skip = data[skip:], max(0, skip - len(data))
                    self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                streamer.sink.remove_client(client)
                logger.info(f"Client {self.client_address[0]} disconnected.")

        def log_message(self, format, *args):
            logger.debug(format % args)

    return Handler


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stream generated chunks as live MPEG-TS over HTTP.")
    parser.add_argument("--stream_dir", type=Path, default=Path("/dev/shm/h3_stream"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20123)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--anchor_image", default=None, help="Shown until the first chunk arrives.")
    parser.add_argument("--preroll_chunks", type=int, default=1, help="Chunks to buffer before playback starts.")
    parser.add_argument("--keep_chunks", action="store_true", help="Do not delete chunk files after loading.")
    parser.add_argument("--metrics_path", type=Path, default=None, help="Append per-chunk metrics as JSONL.")
    parser.add_argument("--record", type=Path, default=None, help="Also write the TS stream to this file.")
    parser.add_argument("--crf", type=int, default=23, help="x264 quality (lower is better and bigger).")
    parser.add_argument(
        "--max_kbps", type=int, default=3000, help="Peak video bitrate cap in kbit/s (0 disables the cap)."
    )
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--exit_when_done", action="store_true", help="Exit after the generator's last chunk plays.")
    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Start with the overlay (buffered frames, latest input) on. "
            "The prompt console's /overlay switches it at runtime."
        ),
    )
    parser.add_argument(
        "--web",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Serve the browser player and prompt box at / (a second, fragmented-MP4 encode).",
    )
    args = parser.parse_args(argv)

    reset_stream_dir(args.stream_dir)
    if args.metrics_path and args.metrics_path.exists():
        args.metrics_path.unlink()
    streamer = Streamer(
        stream_dir=args.stream_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        sample_rate=args.sample_rate,
        anchor_image=args.anchor_image,
        preroll_chunks=args.preroll_chunks,
        delete_chunks=not args.keep_chunks,
        metrics_path=args.metrics_path,
        record_path=args.record,
        crf=args.crf,
        max_kbps=args.max_kbps,
        preset=args.preset,
        overlay=args.overlay,
        web=args.web,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(streamer))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    streamer.start()
    page = f", web page at http://{args.host}:{args.port}/" if args.web else ""
    logger.info(
        f"Streaming http://{args.host}:{args.port}/stream.ts (stats at /stats){page}, reading {args.stream_dir}"
    )

    def shutdown(*_):
        streamer._stop.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    while not streamer._stop.is_set():
        if args.exit_when_done and streamer.finished.is_set():
            break
        time.sleep(0.2)
    streamer.stop()
    server.shutdown()


if __name__ == "__main__":
    main()
