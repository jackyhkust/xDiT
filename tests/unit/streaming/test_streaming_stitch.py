import numpy as np
import pytest

from xfuser.streaming.stitch import ChunkStitcher

FPS = 24
SAMPLE_RATE = 32000


def _chunk(value: int, num_frames: int, audio_value: float | None = 0.0, extra_samples: int = 267):
    frames = np.full((num_frames, 4, 4, 3), value, dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(num_frames)
    if audio_value is None:
        return frames, None
    # The model's soundtrack runs a little longer than its frames.
    samples = round(num_frames * SAMPLE_RATE / FPS) + extra_samples
    return frames, np.full((2, samples), audio_value, dtype=np.float32)


def test_first_chunk_keeps_every_frame_and_later_chunks_drop_frame_zero():
    stitcher = ChunkStitcher(FPS, SAMPLE_RATE)

    first, _ = stitcher.push(*_chunk(10, 124))
    second, _ = stitcher.push(*_chunk(20, 124))

    assert len(first) == 124
    assert len(second) == 123
    assert second[0, 0, 0, 0] == 1
    assert (second[..., 1] == 20).all()
    assert stitcher.emitted_frames == 247


@pytest.mark.parametrize("num_frames", [124, 141, 22])
def test_audio_sample_count_tracks_the_video_timeline_over_many_chunks(num_frames):
    stitcher = ChunkStitcher(FPS, SAMPLE_RATE)
    total_samples = 0
    total_frames = 0

    for index in range(60):
        frames, audio = stitcher.push(*_chunk(index, num_frames))
        total_frames += len(frames)
        total_samples += len(audio)
        # The last frame's audio is held back for the next seam's crossfade.
        assert total_samples == round((total_frames - 1) * SAMPLE_RATE / FPS)

    total_samples += len(stitcher.flush())
    assert total_samples == round(total_frames * SAMPLE_RATE / FPS)


def test_seam_crossfades_from_the_previous_chunk_into_the_next():
    stitcher = ChunkStitcher(FPS, SAMPLE_RATE)
    stitcher.push(*_chunk(0, 124, audio_value=1.0))

    _, audio = stitcher.push(*_chunk(1, 124, audio_value=0.0))

    seam = audio[: round(SAMPLE_RATE / FPS), 0]
    assert seam[0] > 0.99
    assert seam[-1] < 0.01
    assert (np.diff(seam) <= 0).all()


def test_flush_fades_out_and_the_next_chunk_fades_in():
    stitcher = ChunkStitcher(FPS, SAMPLE_RATE)
    stitcher.push(*_chunk(0, 124, audio_value=1.0))

    tail = stitcher.flush()
    _, audio = stitcher.push(*_chunk(1, 124, audio_value=1.0))

    assert tail[0, 0] == pytest.approx(1.0)
    assert tail[-1, 0] == pytest.approx(0.0)
    assert audio[0, 0] == pytest.approx(0.0)
    assert audio[-1, 0] == pytest.approx(1.0)
    assert len(stitcher.flush()) > 0


def test_missing_or_short_audio_becomes_silence():
    stitcher = ChunkStitcher(FPS, SAMPLE_RATE)

    frames, audio = stitcher.push(*_chunk(0, 124, audio_value=None))
    _, short = stitcher.push(*_chunk(1, 124, audio_value=0.5, extra_samples=-20_000))

    assert len(frames) == 124
    assert audio.shape == (round(123 * SAMPLE_RATE / FPS), 2)
    assert not audio.any()
    assert short[-1, 0] == 0.0
    assert short[len(short) // 4, 0] == pytest.approx(0.5)
