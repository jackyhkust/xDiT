"""Pick a stream canvas that follows the anchor image's aspect ratio.

MiniMax-H3 needs height and width divisible by 32 (16x VAE downsampling times
a 2x2 patch). ``fit_resolution`` keeps roughly the requested pixel count, so a
portrait anchor gets a portrait canvas at about the same generation cost.
``cover_resize`` scales an image to cover the canvas and center-crops the
rest, so the picture is never stretched.

Only PIL is imported, so launchers can run this file directly before the
xfuser package (and its GPU checks) is importable::

    python xfuser/streaming/resolution.py anchor.jpg --area 522240   # prints "W H"
"""

from __future__ import annotations

import argparse
import math

from PIL import Image

DEFAULT_AREA = 960 * 544
DEFAULT_DIVISOR = 32


def fit_resolution(
    image_size: tuple[int, int], area: int = DEFAULT_AREA, divisor: int = DEFAULT_DIVISOR
) -> tuple[int, int]:
    """``(width, height)`` with the image's aspect ratio, about ``area`` pixels, both divisible by ``divisor``."""
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size {image_size}.")
    aspect = image_width / image_height
    height = math.sqrt(area / aspect)
    width = height * aspect
    return (
        max(divisor, round(width / divisor) * divisor),
        max(divisor, round(height / divisor) * divisor),
    )


def cover_resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale ``image`` to cover ``size`` (width, height), then center-crop the overflow."""
    width, height = size
    if image.size == (width, height):
        return image
    scale = max(width / image.width, height / image.height)
    scaled = (max(width, round(image.width * scale)), max(height, round(image.height * scale)))
    image = image.resize(scaled, Image.Resampling.LANCZOS)
    left = (scaled[0] - width) // 2
    top = (scaled[1] - height) // 2
    return image.crop((left, top, left + width, top + height))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Print a 'WIDTH HEIGHT' canvas that follows an image's aspect ratio.")
    parser.add_argument("image")
    parser.add_argument("--area", type=int, default=DEFAULT_AREA, help="Target pixel count (default 960x544).")
    parser.add_argument("--divisor", type=int, default=DEFAULT_DIVISOR)
    args = parser.parse_args(argv)
    with Image.open(args.image) as image:
        width, height = fit_resolution(image.size, args.area, args.divisor)
    print(width, height)


if __name__ == "__main__":
    main()
