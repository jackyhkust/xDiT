"""Per-chunk quality-drift metrics for a last-frame-chained stream."""

from __future__ import annotations

import numpy as np

from xfuser.streaming.color import lab_stats


def sharpness(image: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian of the luma; drops as frames blur."""
    luma = image[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    laplacian = luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:] - 4 * luma[1:-1, 1:-1]
    return float(laplacian.var())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))


def chunk_metrics(frames: np.ndarray, anchor: np.ndarray, previous_last: np.ndarray | None) -> dict:
    """Drift metrics for one chunk's raw frames.

    Color and sharpness come from the chunk's last frame, the one the next chunk
    is conditioned on, so a drift that compounds through the chain shows up here
    first. They are reported as differences from the anchor image.
    ``seam_psnr`` compares the conditioning image with the regenerated frame 0.
    A low value means a visible pop at the seam.
    """
    last = frames[-1]
    mean, std = lab_stats(last)
    anchor_mean, anchor_std = lab_stats(anchor)
    anchor_sharpness = sharpness(anchor)
    metrics = {
        "lab_mean": [round(float(v), 3) for v in mean],
        "lab_std": [round(float(v), 3) for v in std],
        "lab_mean_delta": [round(float(v), 3) for v in mean - anchor_mean],
        "lab_std_ratio": [round(float(v), 4) for v in std / np.maximum(anchor_std, 1e-6)],
        "sharpness": round(sharpness(last), 3),
        "sharpness_ratio": round(sharpness(last) / max(anchor_sharpness, 1e-6), 4),
    }
    if previous_last is not None:
        metrics["seam_psnr"] = round(psnr(previous_last, frames[0]), 3)
    return metrics
