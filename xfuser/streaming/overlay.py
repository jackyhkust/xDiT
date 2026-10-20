"""An optional status overlay drawn onto outgoing stream frames.

The overlay is drawn only in the playout process, on the RGB frame just before
it is encoded, so it never feeds back into the generator's conditioning. It
shows how many frames are buffered ahead of playback and the latest prompt the
viewer typed. The prompt console's ``/overlay`` command switches it on and off
while the stream runs.
"""

from __future__ import annotations

import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_MARGIN = 10
_PAD = 8
_LINE_GAP = 4
_WRAP = 44
_MAX_INPUT_LINES = 3
_PANEL_ALPHA = 150


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Overlay:
    def __init__(self, fps: int, font_size: int = 16) -> None:
        self.fps = fps
        self.font = _load_font(font_size)
        self._line_h = font_size + _LINE_GAP

    def lines(self, buffered_frames: int, user_text: str) -> list[str]:
        buffered_s = buffered_frames / self.fps if self.fps else 0
        lines = [f"buffer: {buffered_frames} frames ({buffered_s:.1f}s)"]
        if user_text:
            wrapped = textwrap.wrap(f"input: {user_text}", width=_WRAP) or ["input:"]
            if len(wrapped) > _MAX_INPUT_LINES:
                wrapped = wrapped[:_MAX_INPUT_LINES]
                wrapped[-1] = wrapped[-1][: _WRAP - 3] + "..."
            lines.extend(wrapped)
        else:
            lines.append("input: (none yet)")
        return lines

    def render(self, frame: np.ndarray, *, buffered_frames: int, user_text: str = "") -> np.ndarray:
        lines = self.lines(buffered_frames, user_text)

        base = Image.fromarray(frame).convert("RGBA")
        panel = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)

        widths = [draw.textlength(line, font=self.font) for line in lines]
        box_w = int(max(widths)) + 2 * _PAD
        box_h = self._line_h * len(lines) + 2 * _PAD
        x0, y0 = _MARGIN, _MARGIN
        draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(0, 0, 0, _PANEL_ALPHA))

        y = y0 + _PAD
        for line in lines:
            draw.text((x0 + _PAD, y), line, font=self.font, fill=(255, 255, 255, 255))
            y += self._line_h

        return np.asarray(Image.alpha_composite(base, panel).convert("RGB"))
