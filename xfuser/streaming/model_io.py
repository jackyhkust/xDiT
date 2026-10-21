"""Per-model glue for the streaming generator: chunk media and image conditioning.

The generator loop is the same for every image-to-video model it streams; the
models differ only in how a run returns its frames and audio and in which input
argument carries the conditioning image.

- MiniMax-H3 returns uint8 ``[F, H, W, 3]`` video tensors plus ``output.audio``
  and ``output.audio_sample_rate``, and reads the image from ``input_images``.
- LTX-2.5 returns one namespace (``output.videos[0]``) with float
  ``[B, T, H, W, C]`` frames in ``[0, 1]`` and a vocoder waveform ``[B, C, S]``
  at the vocoder's output rate,
  and reads the image from ``image`` (filled once from ``input_images`` during
  argument preprocessing, so it has to be replaced for every chained chunk).
"""

from __future__ import annotations

import numpy as np
import torch


def chunk_media(model, output) -> tuple[np.ndarray, np.ndarray | None, int]:
    """``(frames, audio, sample_rate)`` of one generated chunk.

    ``frames`` is uint8 ``[F, H, W, 3]``; ``audio`` is float32 ``[channels, samples]`` or None.
    """
    video = output.videos[0]
    if hasattr(video, "frames"):
        # Quantize with torch: multithreaded on the host, unlike a NumPy round of ~10^8 floats.
        frames = torch.from_numpy(np.ascontiguousarray(video.frames[0]))
        frames = frames.mul(255).round_().clamp_(0, 255).to(torch.uint8).numpy()
        # The runner decodes audio outside no_grad, so the vocoder waveform still tracks gradients.
        audio = video.audio[0].detach().float().cpu().numpy() if video.audio is not None else None
        return frames, audio, int(model.pipe.vocoder.config.output_sampling_rate)
    frames = video.numpy()
    audio = output.audio[0].detach().float().cpu().numpy() if output.audio is not None else None
    return frames, audio, int(output.audio_sample_rate)


def with_condition(input_args: dict, image) -> dict:
    """A copy of ``input_args`` that conditions the next run on ``image``."""
    args = dict(input_args, input_images=[image])
    if "image" in input_args:
        args["image"] = image
    return args
