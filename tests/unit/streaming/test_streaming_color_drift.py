import numpy as np
import pytest

from xfuser.streaming.color import lab_stats, lab_to_rgb, match_color_stats, rgb_to_lab
from xfuser.streaming.drift import chunk_metrics, psnr, sharpness


def _image(seed: int = 0, shape=(48, 64)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)


def _blur(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image.astype(np.float64), ((1, 1), (1, 1), (0, 0)), mode="edge")
    out = sum(padded[dy : dy + image.shape[0], dx : dx + image.shape[1]] for dy in range(3) for dx in range(3)) / 9
    return out.round().astype(np.uint8)


def test_lab_round_trip_is_within_one_level():
    image = _image()

    restored = lab_to_rgb(rgb_to_lab(image))

    assert np.abs(restored.astype(int) - image.astype(int)).max() <= 1


def test_lab_of_white_and_black():
    lab = rgb_to_lab(np.array([[[255, 255, 255], [0, 0, 0]]], dtype=np.uint8))

    np.testing.assert_allclose(lab[0, 0], [100.0, 0.0, 0.0], atol=0.01)
    np.testing.assert_allclose(lab[0, 1], [0.0, 0.0, 0.0], atol=0.01)


def test_color_match_moves_statistics_onto_the_reference():
    reference = _image(1)
    tinted = np.clip(reference.astype(int) * [1.0, 0.8, 0.6] + 30, 0, 255).astype(np.uint8)

    matched = match_color_stats(tinted, reference)

    ref_mean, ref_std = lab_stats(reference)
    mean, std = lab_stats(matched)
    tinted_mean, _ = lab_stats(tinted)
    assert np.abs(mean - ref_mean).max() < np.abs(tinted_mean - ref_mean).max() / 4
    np.testing.assert_allclose(std, ref_std, rtol=0.1)


def test_mean_only_color_match_removes_the_cast_but_keeps_the_spread():
    reference = _image(1)
    tinted = np.clip(reference.astype(int) * [1.0, 0.8, 0.6] + 30, 0, 255).astype(np.uint8)

    matched = match_color_stats(tinted, reference, match_std=False)

    ref_mean, _ = lab_stats(reference)
    mean, std = lab_stats(matched)
    tinted_mean, tinted_std = lab_stats(tinted)
    assert np.abs(mean - ref_mean).max() < np.abs(tinted_mean - ref_mean).max() / 4
    np.testing.assert_allclose(std, tinted_std, rtol=0.1)


def test_zero_strength_color_match_leaves_the_image_alone():
    image = _image(2)

    assert np.abs(match_color_stats(image, _image(3), strength=0.0).astype(int) - image.astype(int)).max() <= 1


def test_sharpness_drops_when_the_image_blurs():
    image = _image(4)

    assert sharpness(_blur(image)) < sharpness(image) / 4


def test_chunk_metrics_report_drift_against_the_anchor():
    anchor = np.full((48, 64, 3), 100, dtype=np.uint8)
    anchor[::2] = 140
    brighter = np.clip(anchor.astype(int) + 40, 0, 255).astype(np.uint8)
    frames = np.stack([anchor, brighter])

    first = chunk_metrics(frames, anchor, previous_last=None)
    later = chunk_metrics(frames, anchor, previous_last=anchor)

    assert "seam_psnr" not in first
    assert later["seam_psnr"] == float("inf")
    assert first["lab_mean_delta"][0] > 10
    assert first["sharpness_ratio"] == pytest.approx(1.0, rel=0.05)


def test_psnr_of_a_known_error():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.full((4, 4, 3), 255, dtype=np.uint8)

    assert psnr(a, b) == pytest.approx(0.0)
