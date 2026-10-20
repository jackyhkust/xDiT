import io
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from xfuser.streaming.chunks import Chunk, mark_done, write_chunk

av = pytest.importorskip("av")

pytestmark = pytest.mark.slow

FPS = 24
SAMPLE_RATE = 32000
WIDTH, HEIGHT = 96, 64
CHUNK_FRAMES = 22  # 17 * 1 + 5, the shortest length the H3 VAE decodes


def _chunk(index: int, level: int | None = None) -> Chunk:
    level = 40 + 60 * index if level is None else level
    frames = np.full((CHUNK_FRAMES, HEIGHT, WIDTH, 3), level, dtype=np.uint8)
    samples = round(CHUNK_FRAMES * SAMPLE_RATE / FPS) + 267
    tone = 0.2 * np.sin(np.arange(samples) * 2 * np.pi * 440 / SAMPLE_RATE, dtype=np.float32)
    return Chunk(
        index=index,
        frames=frames,
        audio=np.stack([tone, tone]),
        sample_rate=SAMPLE_RATE,
        fps=FPS,
        meta={"gen_time_s": 0.1},
    )


@pytest.fixture
def serve(tmp_path):
    from xfuser.streaming.ts_server import Streamer, make_handler

    started = []

    def start(**kwargs):
        streamer = Streamer(
            stream_dir=tmp_path / "stream",
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            sample_rate=SAMPLE_RATE,
            record_path=tmp_path / "record.ts",
            metrics_path=tmp_path / "metrics.jsonl",
            **kwargs,
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(streamer))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        streamer.start()
        started.append((streamer, server))
        return streamer, f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for streamer, server in started:
        streamer.stop()
        server.shutdown()


def _read_stream(url: str, into: io.BytesIO) -> threading.Thread:
    def read():
        with urllib.request.urlopen(f"{url}/stream.ts", timeout=30) as response:
            while chunk := response.read(65536):
                into.write(chunk)

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    return thread


def _decode(data: bytes):
    container = av.open(io.BytesIO(data), format="mpegts")
    video = container.streams.video[0]
    frames = [
        (frame.pts * video.time_base, int(frame.to_ndarray(format="rgb24")[..., 0].mean()))
        for frame in container.decode(video=0)
    ]
    container.close()
    container = av.open(io.BytesIO(data), format="mpegts")
    audio_samples = sum(frame.samples for frame in container.decode(audio=0))
    container.close()
    return frames, audio_samples


def _assert_continuous_24fps(times):
    steps = np.diff([float(t) for t in times])
    np.testing.assert_allclose(steps, 1 / FPS, atol=1e-3)


def test_chunks_play_back_to_back_at_24_fps_with_audio(serve, tmp_path):
    for index in range(3):
        write_chunk(tmp_path / "stream", _chunk(index))
    mark_done(tmp_path / "stream", 3)

    streamer, url = serve()
    received = io.BytesIO()
    client = _read_stream(url, received)
    assert streamer.finished.wait(timeout=30)
    client.join(timeout=10)

    summary = streamer.summary()
    assert summary["stall_frames"] == 0
    assert summary["chunks_played"] == 3
    # The first chunk is shown whole; later chunks drop their duplicated frame 0.
    assert summary["content_frames"] == CHUNK_FRAMES + 2 * (CHUNK_FRAMES - 1)

    recorded, audio_samples = _decode((tmp_path / "record.ts").read_bytes())
    assert len(recorded) == summary["frames_emitted"]
    _assert_continuous_24fps([t for t, _ in recorded])
    assert abs(audio_samples / SAMPLE_RATE - len(recorded) / FPS) < 0.1
    levels = [level for _, level in recorded]
    assert levels[-1] == pytest.approx(40 + 60 * 2, abs=4)

    client_frames, client_audio = _decode(received.getvalue())
    assert len(client_frames) >= summary["content_frames"]
    assert client_audio > 0
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 3


def test_an_underrun_repeats_frames_without_breaking_timestamps(serve, tmp_path):
    write_chunk(tmp_path / "stream", _chunk(0))
    streamer, _ = serve()

    deadline = time.monotonic() + 15
    while streamer.summary()["chunks_played"] < 1 and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.5)
    write_chunk(tmp_path / "stream", _chunk(1))
    mark_done(tmp_path / "stream", 2)
    assert streamer.finished.wait(timeout=30)

    summary = streamer.summary()
    assert summary["stall_frames"] >= FPS // 4
    assert summary["content_frames"] == 2 * CHUNK_FRAMES - 1
    recorded, _ = _decode((tmp_path / "record.ts").read_bytes())
    _assert_continuous_24fps([t for t, _ in recorded])


def test_a_cut_drops_buffered_frames_and_the_next_chunk_continues_from_the_cut(serve, tmp_path):
    from xfuser.streaming.chunks import read_position, write_cut

    stream = tmp_path / "stream"
    for index in range(4):
        write_chunk(stream, _chunk(index))
    streamer, _ = serve()
    deadline = time.monotonic() + 15
    while streamer.summary()["chunks_loaded"] < 4 and time.monotonic() < deadline:
        time.sleep(0.01)

    keep_through = 30
    write_cut(stream, epoch=1, keep_through=keep_through, after_chunk=3, prompt_id=1)
    while not streamer.summary()["cuts"] and time.monotonic() < deadline:
        time.sleep(0.01)
    write_chunk(stream, _chunk(4, level=10))
    mark_done(stream, 5)
    assert streamer.finished.wait(timeout=30)

    summary = streamer.summary()
    assert summary["cuts"][0]["late_frames"] == 0
    assert summary["cuts"][0]["dropped_s"] > 1.0
    assert summary["stall_frames"] == 0
    assert summary["chunks_played"] == 5
    assert summary["content_frames"] == keep_through + 1 + CHUNK_FRAMES - 1
    assert read_position(stream) == summary["content_frames"]

    recorded, _ = _decode((tmp_path / "record.ts").read_bytes())
    _assert_continuous_24fps([t for t, _ in recorded])
    levels = [level for _, level in recorded]
    assert levels[-1] == pytest.approx(10, abs=4)
    # Frame 30 sits in chunk 1 (frames 22-42), so chunks 2 and 3 never play.
    assert not any(abs(level - 160) <= 4 or abs(level - 220) <= 4 for level in levels)


def test_the_overlay_switch_turns_the_overlay_on_mid_stream(serve, tmp_path):
    from xfuser.streaming.prompt_queue import read_overlay, toggle_overlay

    stream = tmp_path / "stream"
    for index in range(4):
        write_chunk(stream, _chunk(index, level=200))
    mark_done(stream, 4)
    streamer, _ = serve()
    assert read_overlay(stream) is False

    deadline = time.monotonic() + 15
    while streamer.summary()["content_frames"] < 12 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert toggle_overlay(stream) is True
    assert streamer.finished.wait(timeout=30)
    assert streamer.summary()["overlay"] is True

    container = av.open(str(tmp_path / "record.ts"), format="mpegts")
    # A pixel inside the panel's padding: plain frame level before, darkened by the panel after.
    corner = [int(frame.to_ndarray(format="rgb24")[13, 13, 0]) for frame in container.decode(video=0)]
    container.close()
    # The first emitted frames can be lead-in black, so look at the start of the content.
    assert max(corner[:10]) == pytest.approx(200, abs=12)
    assert corner[-1] < 140


def test_a_client_joining_mid_stream_decodes_within_one_gop(serve, tmp_path):
    for index in range(4):
        write_chunk(tmp_path / "stream", _chunk(index))
    mark_done(tmp_path / "stream", 4)
    streamer, url = serve()

    deadline = time.monotonic() + 15
    while streamer.summary()["frames_emitted"] < 30 and time.monotonic() < deadline:
        time.sleep(0.01)
    joined_at = streamer.summary()["frames_emitted"]
    received = io.BytesIO()
    client = _read_stream(url, received)
    assert streamer.finished.wait(timeout=30)
    client.join(timeout=10)

    frames, _ = _decode(received.getvalue())
    remaining = streamer.summary()["frames_emitted"] - joined_at
    assert len(frames) >= remaining - FPS - 2
