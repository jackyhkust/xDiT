import numpy as np

from xfuser.streaming.overlay import Overlay


def test_overlay_preserves_shape_and_dtype():
    overlay = Overlay(fps=24)
    frame = np.full((544, 960, 3), 30, dtype=np.uint8)

    out = overlay.render(frame, buffered_frames=120, user_text="make the cat fly over the neon city")

    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_overlay_only_touches_the_top_left_panel():
    overlay = Overlay(fps=24)
    frame = np.zeros((544, 960, 3), dtype=np.uint8)

    out = overlay.render(frame, buffered_frames=48, user_text="a dog on a surfboard")

    # The panel sits in the top-left corner; the opposite corner is untouched.
    assert out[:220, :420].sum() > 0
    assert out[300:, 500:].sum() == 0


def test_overlay_does_not_modify_the_input_frame():
    frame = np.zeros((256, 256, 3), dtype=np.uint8)

    Overlay(fps=24).render(frame, buffered_frames=10, user_text="x")

    assert frame.sum() == 0


def test_overlay_lines_show_buffer_and_latest_input():
    overlay = Overlay(fps=24)

    assert overlay.lines(48, "") == ["buffer: 48 frames (2.0s)", "input: (none yet)"]
    assert overlay.lines(0, "a red kite") == ["buffer: 0 frames (0.0s)", "input: a red kite"]


def test_overlay_caps_a_long_input_at_three_lines():
    lines = Overlay(fps=24).lines(0, "word " * 200)

    assert len(lines) == 1 + 3
    assert lines[-1].endswith("...")
