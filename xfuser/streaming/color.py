"""sRGB <-> CIELAB (D65) conversion and Reinhard color-statistics transfer, in NumPy."""

from __future__ import annotations

import numpy as np

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_EPSILON = 216 / 24389
_KAPPA = 24389 / 27


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """uint8 ``[..., 3]`` sRGB to float ``[..., 3]`` L*a*b*."""
    rgb = image.astype(np.float64) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _RGB_TO_XYZ.T / _WHITE_D65
    f = np.where(xyz > _EPSILON, np.cbrt(xyz), (_KAPPA * xyz + 16) / 116)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Float ``[..., 3]`` L*a*b* to uint8 ``[..., 3]`` sRGB, clipped to gamut."""
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    xyz = np.where(f**3 > _EPSILON, f**3, (116 * f - 16) / _KAPPA) * _WHITE_D65
    linear = np.clip(xyz @ _XYZ_TO_RGB.T, 0.0, 1.0)
    rgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def lab_stats(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel L*a*b* mean and standard deviation of a uint8 RGB image."""
    lab = rgb_to_lab(image).reshape(-1, 3)
    return lab.mean(axis=0), lab.std(axis=0)


def match_color_stats(
    image: np.ndarray, reference: np.ndarray, strength: float = 1.0, match_std: bool = True
) -> np.ndarray:
    """Reinhard transfer: move ``image``'s per-channel L*a*b* mean (and std) toward ``reference``'s.

    ``strength`` blends between the input (0) and the full transfer (1). With
    ``match_std=False`` only the means shift: a global cast is removed, but no
    channel's spread is stretched. Repeated every chunk, stretching amplifies
    whatever local outlier the model leaves behind, such as an edge band.
    """
    lab = rgb_to_lab(image)
    mean, std = lab.reshape(-1, 3).mean(axis=0), lab.reshape(-1, 3).std(axis=0)
    ref_mean, ref_std = lab_stats(reference)
    scale = ref_std / np.maximum(std, 1e-6) if match_std else 1.0
    matched = (lab - mean) * scale + ref_mean
    return lab_to_rgb(lab + strength * (matched - lab))
