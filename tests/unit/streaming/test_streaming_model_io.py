from types import SimpleNamespace

import numpy as np
import torch

from xfuser.model_executor.models.runner_models.base_model import DiffusionOutput
from xfuser.streaming.model_io import chunk_media, with_condition


def test_h3_output_passes_through():
    frames = torch.randint(0, 256, (5, 4, 6, 3), dtype=torch.uint8)
    audio = torch.rand(1, 2, 100, dtype=torch.bfloat16)
    output = SimpleNamespace(videos=[frames], audio=audio, audio_sample_rate=32000)

    got_frames, got_audio, rate = chunk_media(SimpleNamespace(), output)

    assert got_frames.dtype == np.uint8 and np.array_equal(got_frames, frames.numpy())
    assert got_audio.dtype == np.float32 and got_audio.shape == (2, 100)
    assert rate == 32000


def test_h3_output_without_audio():
    output = SimpleNamespace(videos=[torch.zeros(2, 4, 6, 3, dtype=torch.uint8)], audio=None, audio_sample_rate=32000)

    _, audio, _ = chunk_media(SimpleNamespace(), output)

    assert audio is None


def test_ltx_output_is_quantized_and_uses_vocoder_rate():
    frames = np.array([0.0, 0.5, 1.0, 1.2, -0.1], dtype=np.float32).reshape(1, 5, 1, 1, 1).repeat(3, axis=-1)
    waveform = torch.rand(1, 2, 480, dtype=torch.bfloat16, requires_grad=True)
    output = DiffusionOutput(videos=SimpleNamespace(frames=frames, audio=waveform), pipe_args={})
    vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=48000))
    model = SimpleNamespace(pipe=SimpleNamespace(vocoder=vocoder))

    got_frames, got_audio, rate = chunk_media(model, output)

    assert got_frames.dtype == np.uint8 and got_frames.shape == (5, 1, 1, 3)
    assert got_frames[:, 0, 0, 0].tolist() == [0, 128, 255, 255, 0]
    assert got_audio.dtype == np.float32 and got_audio.shape == (2, 480)
    assert rate == 48000


def test_with_condition_replaces_every_image_argument():
    anchor, frame = object(), object()
    h3_args = {"input_images": [anchor], "prompt": "p"}
    ltx_args = {"input_images": [anchor], "image": anchor, "prompt": "p"}

    assert with_condition(h3_args, frame) == {"input_images": [frame], "prompt": "p"}
    assert with_condition(ltx_args, frame) == {"input_images": [frame], "image": frame, "prompt": "p"}
    assert h3_args["input_images"] == [anchor] and ltx_args["image"] is anchor
