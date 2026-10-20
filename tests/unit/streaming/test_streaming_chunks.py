import numpy as np

from xfuser.streaming.chunks import (
    Chunk,
    chunk_path,
    mark_done,
    read_chunk,
    read_done,
    read_played,
    reset_stream_dir,
    write_chunk,
    write_played,
)


def _chunk(index: int, with_audio: bool = True) -> Chunk:
    frames = np.random.default_rng(index).integers(0, 256, size=(5, 8, 12, 3), dtype=np.uint8)
    audio = np.linspace(-1, 1, 2 * 100, dtype=np.float32).reshape(2, 100) if with_audio else None
    return Chunk(index=index, frames=frames, audio=audio, sample_rate=32000, fps=24, meta={"gen_time_s": 3.5})


def test_chunk_round_trip(tmp_path):
    chunk = _chunk(3)

    write_chunk(tmp_path, chunk)
    loaded = read_chunk(tmp_path, 3)

    np.testing.assert_array_equal(loaded.frames, chunk.frames)
    np.testing.assert_array_equal(loaded.audio, chunk.audio)
    assert (loaded.sample_rate, loaded.fps, loaded.meta) == (32000, 24, {"gen_time_s": 3.5})
    assert list(tmp_path.iterdir()) == [chunk_path(tmp_path, 3)]


def test_chunk_without_audio_round_trips(tmp_path):
    write_chunk(tmp_path, _chunk(0, with_audio=False))

    assert read_chunk(tmp_path, 0).audio is None


def test_a_missing_chunk_reads_as_none_and_delete_removes_the_file(tmp_path):
    write_chunk(tmp_path, _chunk(0))

    assert read_chunk(tmp_path, 1) is None
    assert read_chunk(tmp_path, 0, delete=True) is not None
    assert read_chunk(tmp_path, 0) is None


def test_played_and_done_markers(tmp_path):
    assert read_played(tmp_path) == 0
    assert read_done(tmp_path) is None

    write_played(tmp_path, 7)
    mark_done(tmp_path, 12)

    assert read_played(tmp_path) == 7
    assert read_done(tmp_path) == 12


def test_reset_removes_only_stream_files(tmp_path):
    write_chunk(tmp_path, _chunk(0))
    write_played(tmp_path, 1)
    mark_done(tmp_path, 1)
    (tmp_path / "keep.txt").write_text("x")

    reset_stream_dir(tmp_path)

    assert [path.name for path in tmp_path.iterdir()] == ["keep.txt"]
