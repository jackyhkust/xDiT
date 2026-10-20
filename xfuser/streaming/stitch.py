"""Join last-frame-chained I2V chunks into one continuous audio/video timeline."""

from __future__ import annotations

import numpy as np


def _equal_power_crossfade(tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    ramp = (np.arange(len(tail), dtype=np.float32) + 0.5) / len(tail) * (np.pi / 2)
    return tail * np.cos(ramp)[:, None] + head * np.sin(ramp)[:, None]


def _take(audio: np.ndarray, start: int, count: int) -> np.ndarray:
    """``audio[start:start + count]``, zero-padded where the chunk's audio runs short."""
    out = np.zeros((count, audio.shape[1]), dtype=np.float32)
    available = audio[start : start + count]
    out[: len(available)] = available
    return out


class ChunkStitcher:
    """Turn generated chunks into a gapless timeline of frames and samples.

    Chunk N+1 is conditioned on the last frame of chunk N and regenerates it as
    its own frame 0, so frame 0 of every chunk after the first is dropped.

    Audio is cut on the video timeline rather than taken whole: emitted frame
    ``j`` owns samples ``[round(j * sr / fps), round((j + 1) * sr / fps))``. The
    count stays exact over any number of chunks, so A/V never drifts apart, even
    though the model's soundtrack is slightly longer than its frames.

    The audio under a dropped frame 0 covers the same moment as the previous
    chunk's last frame. The two are equal-power crossfaded instead of one being
    thrown away. That is why the last frame's audio is held back until the next
    chunk arrives, or until ``flush`` releases it with a fade-out on underrun.
    """

    def __init__(self, fps: int, sample_rate: int, channels: int = 2) -> None:
        self.fps = fps
        self.sample_rate = sample_rate
        self.channels = channels
        self.emitted_frames = 0
        self._tail: np.ndarray | None = None
        self._fade_in_next = False

    def _boundary(self, frame: int) -> int:
        return round(frame * self.sample_rate / self.fps)

    def _samples_per_frame(self, frame: int) -> int:
        return self._boundary(frame + 1) - self._boundary(frame)

    def push(self, frames: np.ndarray, audio: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Stitch one chunk. ``audio`` is ``[channels, samples]`` or None for silence.

        Returns the frames to show and the float32 ``[samples, channels]`` audio
        to play after everything returned so far.
        """
        if audio is None:
            audio = np.zeros((self.channels, self._boundary(len(frames))), dtype=np.float32)
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).T)

        first_kept = 1 if self.emitted_frames > 0 else 0
        start = self.emitted_frames
        end = start + len(frames) - first_kept
        pieces = []
        cursor = 0

        if self._tail is not None:
            head = _take(audio, 0, len(self._tail))
            pieces.append(_equal_power_crossfade(self._tail, head))
            cursor = len(self._tail)
        elif first_kept:
            # The previous last frame's audio was flushed; skip its overlap here.
            cursor = self._samples_per_frame(0)

        body_count = self._boundary(end - 1) - self._boundary(start)
        body = _take(audio, cursor, body_count)
        if self._fade_in_next and len(body):
            fade = min(len(body), self._samples_per_frame(start))
            body[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
        pieces.append(body)
        cursor += body_count

        self._tail = _take(audio, cursor, self._samples_per_frame(end - 1))
        self._fade_in_next = False
        self.emitted_frames = end
        return frames[first_kept:], np.concatenate(pieces, axis=0)

    def rewind(self, emitted_frames: int) -> None:
        """Continue the timeline after frame ``emitted_frames - 1`` (a cut-in dropped the frames after it).

        The held-back seam audio belonged to a dropped frame, so it is discarded and
        the next chunk's audio fades in instead of being crossfaded.
        """
        self.emitted_frames = emitted_frames
        self._tail = None
        self._fade_in_next = True

    def flush(self) -> np.ndarray:
        """Release the held-back last-frame audio with a fade-out (use on underrun)."""
        if self._tail is None:
            return np.zeros((0, self.channels), dtype=np.float32)
        tail = self._tail * np.linspace(1.0, 0.0, len(self._tail), dtype=np.float32)[:, None]
        self._tail = None
        self._fade_in_next = True
        return tail
