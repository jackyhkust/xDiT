import numpy as np
import pytest
from PIL import Image

from xfuser.streaming.chunks import read_cut, read_position, reset_stream_dir, write_cut, write_position
from xfuser.streaming.history import FrameHistory
from xfuser.streaming.resolution import cover_resize, fit_resolution
from xfuser.streaming.stitch import ChunkStitcher


@pytest.mark.parametrize(
    "image_size, expected",
    [
        ((832, 1104), (640, 832)),  # the portrait cat anchor
        ((1920, 1080), (960, 544)),
        ((1024, 1024), (736, 736)),
    ],
)
def test_fit_resolution_follows_the_aspect_ratio_on_the_32_grid(image_size, expected):
    width, height = fit_resolution(image_size)
    assert (width, height) == expected
    assert width % 32 == 0 and height % 32 == 0


def test_cover_resize_crops_instead_of_stretching():
    image = np.zeros((100, 50, 3), dtype=np.uint8)
    image[:, :25] = 255
    out = cover_resize(Image.fromarray(image), (40, 40))
    assert out.size == (40, 40)
    row = np.asarray(out)[20, :, 0]
    assert row[:18].min() > 200 and row[-18:].max() < 50


def test_frame_history_indexes_truncates_and_prunes():
    history = FrameHistory()
    history.append(np.arange(10))
    history.append(np.arange(10, 19))
    history.append(np.arange(19, 28))
    assert history.end == 28
    assert history.frame(0) == 0 and history.frame(15) == 15 and history.frame(27) == 27

    history.truncate(12)
    assert history.end == 13
    assert history.frame(12) == 12
    with pytest.raises(IndexError):
        history.frame(13)

    history.append(np.arange(100, 105))
    assert history.frame(13) == 100
    history.prune(played=11)
    with pytest.raises(IndexError):
        history.frame(5)
    assert history.frame(12) == 12


def test_position_and_cut_markers_round_trip_and_reset(tmp_path):
    assert read_position(tmp_path) == 0 and read_cut(tmp_path) is None
    write_position(tmp_path, 321)
    write_cut(tmp_path, epoch=2, keep_through=400, after_chunk=7, prompt_id=3)
    assert read_position(tmp_path) == 321
    assert read_cut(tmp_path) == {"epoch": 2, "keep_through": 400, "after_chunk": 7, "prompt_id": 3}
    reset_stream_dir(tmp_path)
    assert read_position(tmp_path) == 0 and read_cut(tmp_path) is None


def test_stitcher_rewind_continues_the_timeline_from_the_cut():
    fps, sr = 24, 32000
    stitcher = ChunkStitcher(fps, sr)
    frames = np.zeros((22, 4, 4, 3), dtype=np.uint8)
    audio = np.ones((2, round(22 * sr / fps) + 100), dtype=np.float32)
    stitcher.push(frames, audio)
    stitcher.push(frames, audio)
    stitcher.rewind(30)

    shown, samples = stitcher.push(frames, audio)
    assert len(shown) == 21  # still chained: frame 0 duplicates the cut frame
    assert stitcher.emitted_frames == 51
    expected = round(50 * sr / fps) - round(30 * sr / fps)
    assert len(samples) == expected
    assert samples[0, 0] == pytest.approx(0.0, abs=1e-3)  # fades in, no crossfade with dropped audio


def test_audio_buffer_truncate_keeps_the_head_and_fades_out():
    from xfuser.streaming.ts_server import AudioBuffer

    buffer = AudioBuffer(channels=2)
    buffer.append(np.ones((100, 2), dtype=np.float32))
    buffer.append(np.ones((100, 2), dtype=np.float32))
    buffer.truncate(150, fade=10)
    assert buffer.available == 150
    out = buffer.read(150)
    assert out[:140].min() == 1.0
    assert out[-1, 0] == pytest.approx(0.0)
