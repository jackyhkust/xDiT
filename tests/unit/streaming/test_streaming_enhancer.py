import numpy as np
import pytest
import torch

from xfuser.streaming.enhancer import I2VA_HEADER, PromptEnhancer, parse_enhanced

_GOOD = """\
Sure, here you go.
===TRANSITION===
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] the cat begins to lift off gently.

overall_soundscape: soft wind.

non_diegetic_music: light synth.
===CONTINUATION===
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] the cat soars over the city.

overall_soundscape: rushing air.

non_diegetic_music: soaring strings.
"""


def test_parse_splits_transition_and_continuation():
    result = parse_enhanced(_GOOD, current="old", user_text="fly")
    assert result.ok
    assert "lift off gently" in result.transition
    assert "soars over the city" in result.continuation
    assert "===CONTINUATION===" not in result.transition


def test_parse_falls_back_when_markers_missing():
    result = parse_enhanced("no markers here", current="a quiet beach", user_text="make it rain")
    assert not result.ok
    assert result.transition == result.continuation
    assert "make it rain" in result.transition
    assert "a quiet beach" in result.transition


def test_parse_reuses_the_transition_when_output_is_cut_off():
    result = parse_enhanced("===TRANSITION===\njust one section", current="c", user_text="u")
    assert not result.ok
    assert result.transition == result.continuation
    assert result.transition.startswith(I2VA_HEADER)
    assert "just one section" in result.transition


def test_parse_adds_the_i2va_header_to_compact_sections():
    compact = (
        "===TRANSITION===\nintegrated_multimodal_description: [Shot 1] snow starts.\n"
        "overall_soundscape: wind.\nnon_diegetic_music: piano.\n"
        "===CONTINUATION===\nintegrated_multimodal_description: [Shot 1] a snowy beach.\n"
        "overall_soundscape: wind.\nnon_diegetic_music: piano.\n"
    )
    result = parse_enhanced(compact, current="c", user_text="snow")
    assert result.ok
    for prompt in (result.transition, result.continuation):
        assert prompt.startswith(I2VA_HEADER + "\n\nintegrated_multimodal_description: [Shot 1]")
    # Sections that already carry the header are not given a second one.
    assert parse_enhanced(_GOOD, current="old", user_text="fly").transition.count(I2VA_HEADER) == 1


def test_static_fallback_without_user_text_keeps_current():
    result = PromptEnhancer.fallback("current scene", "")
    assert result.transition == "current scene"
    assert not result.ok


class _FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<chat>"

    def __call__(self, text=None, images=None, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    def batch_decode(self, tokens, skip_special_tokens=True):
        return [_GOOD]


class _FakeTextEncoder:
    def __init__(self):
        self.dtype = torch.float32
        self.lm_head = object()  # not None, so no reload happens
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._p])

    def generate(self, **kwargs):
        # Echo the prompt tokens plus three "generated" tokens.
        input_ids = kwargs["input_ids"]
        extra = torch.tensor([[7, 8, 9]])
        return torch.cat([input_ids, extra], dim=1)


class _FakePipe:
    def __init__(self):
        self.text_encoder = _FakeTextEncoder()
        self.processor = _FakeProcessor()


def test_enhance_end_to_end_with_fake_model():
    enhancer = PromptEnhancer(_FakePipe(), model_name="fake", text_encoder_tp_degree=1)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result = enhancer.enhance("orig", "current", "fly over the city", frame)
    assert result.ok
    assert "soars over the city" in result.continuation


def test_tp_degree_above_one_is_rejected():
    with pytest.raises(ValueError):
        PromptEnhancer(_FakePipe(), model_name="fake", text_encoder_tp_degree=8)
