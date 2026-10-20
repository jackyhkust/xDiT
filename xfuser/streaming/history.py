"""The generator's copy of frames that may still be unplayed, indexed on the stream timeline.

Global frame ``i`` is the ``i``-th content frame the streamer will show (chained
chunks contribute their frames after the duplicated frame 0). A cut-in picks a
frame from here as the next chunk's conditioning image.
"""

from __future__ import annotations

import collections

import numpy as np


class FrameHistory:
    def __init__(self) -> None:
        self._chunks: collections.deque[tuple[int, np.ndarray]] = collections.deque()
        self.end = 0

    def append(self, frames: np.ndarray) -> None:
        self._chunks.append((self.end, frames))
        self.end += len(frames)

    def frame(self, index: int) -> np.ndarray:
        for start, frames in self._chunks:
            if start <= index < start + len(frames):
                return frames[index - start]
        raise IndexError(f"Frame {index} is not in the history ({self.end} frames generated).")

    def truncate(self, keep_through: int) -> None:
        """Forget frames after ``keep_through``; generation continues from there."""
        while self._chunks and self._chunks[-1][0] > keep_through:
            self._chunks.pop()
        if self._chunks:
            start, frames = self._chunks[-1]
            self._chunks[-1] = (start, frames[: keep_through + 1 - start])
        self.end = keep_through + 1

    def prune(self, played: int) -> None:
        """Drop chunks that have fully played, always keeping the newest one."""
        while len(self._chunks) > 1 and self._chunks[0][0] + len(self._chunks[0][1]) <= played:
            self._chunks.popleft()
