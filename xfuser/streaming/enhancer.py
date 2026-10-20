"""Rewrite interactive user prompts with MiniMax-H3's own Qwen3-VL conditioner.

MiniMax-H3 conditions on a Qwen3-VL text encoder that is already resident on the
GPU. xDiT drops its ``lm_head`` after loading (the diffusion path only needs
hidden states), so this module reloads ``lm_head.weight`` from the checkpoint and
runs the conditioner as a normal vision-language model to rewrite prompts.

Given the original prompt, the current prompt, a new user request and the last
generated frame, it returns two prompts in MiniMax-H3's I2VA format: a
``TRANSITION`` prompt that eases from the current scene toward the request, and a
``CONTINUATION`` prompt that sustains the new scene. Both start from the last
frame (Picture 1), so the stream never cuts.

Only the writer rank runs generation; the caller broadcasts the result. Text
encoder tensor parallelism (``--text_encoder_tp_degree > 1``) shards the
conditioner and cannot run a standalone ``generate``, so enhancement requires
``text_encoder_tp_degree <= 1`` (the default).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from xfuser.core.utils.runner_utils import log

_TRANSITION_MARKER = "===TRANSITION==="
_CONTINUATION_MARKER = "===CONTINUATION==="

SYSTEM_PROMPT = """\
You rewrite prompts for the MiniMax-H3 image-to-video (I2VA) model, which turns a \
first frame plus a prompt into a continuous video with audio. You are given the \
original stream prompt, the prompt currently rendering, a new request from a \
viewer, and Picture 1: the frame the next clip starts from. Rewrite the request into I2VA prompts that change the scene toward the \
request without ever cutting.

Rules for continuity:
- Picture 1 is the actual first frame at 0.00 s and belongs to [Shot 1]. Anchor \
the subject identity, clothing, colors, key objects, composition, style, lighting \
and camera from Picture 1, then develop the action forward.
- One continuous shot, no cuts, no scene changes, no camera teleports.
- Introduce the requested change gradually and plausibly from what is on screen.

Be concise: the stream waits while you write. Output exactly these two sections, \
with no preamble, explanation or anything after the second section:

===TRANSITION===
integrated_multimodal_description: [Shot 1] <at most 60 words: anchor on Picture 1, then the gradual change toward the request>
overall_soundscape: <at most 12 words>
non_diegetic_music: <at most 8 words>

===CONTINUATION===
integrated_multimodal_description: [Shot 1] <at most 60 words: the requested scene, sustained and developing>
overall_soundscape: <at most 12 words>
non_diegetic_music: <at most 8 words>
"""

# Every I2VA prompt starts with this fixed reference line; it is added in code so
# the model does not spend tokens (and stream time) repeating it.
I2VA_HEADER = (
    "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
)


def _with_header(section: str) -> str:
    if section.lstrip().startswith("For the target video"):
        return section
    return f"{I2VA_HEADER}\n\n{section}"


@dataclass
class EnhancedPrompt:
    transition: str
    continuation: str
    raw: str
    ok: bool


def _naive_prompt(current: str, user_text: str) -> str:
    """Fallback used when the enhancer is off or its output cannot be parsed."""
    user_text = user_text.strip()
    if not user_text:
        return current
    return f"{current.strip()}\n\nNow, gradually and without any cut: {user_text}"


def parse_enhanced(text: str, current: str, user_text: str) -> EnhancedPrompt:
    """Split the model output into TRANSITION and CONTINUATION, with a fallback."""
    transition = continuation = None
    if _TRANSITION_MARKER in text:
        after = text.split(_TRANSITION_MARKER, 1)[1]
        if _CONTINUATION_MARKER in after:
            transition_part, continuation_part = after.split(_CONTINUATION_MARKER, 1)
            transition = transition_part.strip()
            continuation = continuation_part.strip()
        else:
            transition = after.strip()
    # Trim anything after a trailing marker repetition.
    if continuation:
        continuation = re.split(r"\n=+[A-Z]+=+", continuation)[0].strip()
    if transition and continuation:
        return EnhancedPrompt(
            transition=_with_header(transition), continuation=_with_header(continuation), raw=text, ok=True
        )
    if transition:
        # Output was cut off before the CONTINUATION section; the transition still
        # describes the requested change, so keep using it.
        prompt = _with_header(transition)
        return EnhancedPrompt(transition=prompt, continuation=prompt, raw=text, ok=False)
    fallback = _naive_prompt(current, user_text)
    return EnhancedPrompt(transition=fallback, continuation=fallback, raw=text, ok=False)


class PromptEnhancer:
    def __init__(
        self,
        pipe,
        model_name: str,
        text_encoder_tp_degree: int = 1,
        max_new_tokens: int = 384,
    ) -> None:
        if text_encoder_tp_degree and text_encoder_tp_degree > 1:
            raise ValueError(
                "Prompt enhancement needs the full Qwen3-VL conditioner on one rank, "
                "but --text_encoder_tp_degree > 1 shards it. Use --text_encoder_tp_degree 1 "
                "(the default) with --prompt_enhancer qwen3vl, or run with --prompt_enhancer none."
            )
        self.text_encoder = pipe.text_encoder
        self.processor = pipe.processor
        self.max_new_tokens = max_new_tokens
        self._device = next(self.text_encoder.parameters()).device
        self._ensure_lm_head(model_name)

    def _ensure_lm_head(self, model_name: str) -> None:
        """Restore the lm_head that xDiT drops after loading the conditioner."""
        if getattr(self.text_encoder, "lm_head", None) is not None:
            return
        from safetensors.torch import load_file
        from transformers.utils import cached_file

        index = cached_file(model_name, "model.safetensors.index.json", subfolder="text_encoder")
        import json
        from pathlib import Path

        shard_name = json.loads(Path(index).read_text())["weight_map"]["lm_head.weight"]
        shard_path = cached_file(model_name, shard_name, subfolder="text_encoder")
        weight = load_file(shard_path)["lm_head.weight"]

        config = self.text_encoder.config.get_text_config()
        lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        lm_head.weight = torch.nn.Parameter(weight.to(self._device, dtype=self.text_encoder.dtype))
        self.text_encoder.lm_head = lm_head.to(self._device)
        log("Reloaded Qwen3-VL lm_head for prompt enhancement.")

    def _build_inputs(self, original: str, current: str, user_text: str, image: Image.Image):
        instruction = (
            f"Original stream prompt:\n{original.strip()}\n\n"
            f"Currently rendering:\n{current.strip()}\n\n"
            f"Viewer request:\n{user_text.strip()}\n\n"
            "Picture 1, shown above, is the frame the next clip starts from. Rewrite the request now."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": instruction}],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        return {key: value.to(self._device) for key, value in inputs.items()}

    @torch.no_grad()
    def enhance(self, original: str, current: str, user_text: str, last_frame: np.ndarray) -> EnhancedPrompt:
        image = Image.fromarray(np.asarray(last_frame).astype("uint8")).convert("RGB")
        inputs = self._build_inputs(original, current, user_text, image)
        start = time.perf_counter()
        generated = self.text_encoder.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        elapsed = time.perf_counter() - start
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        count = new_tokens.shape[1]
        log(
            f"Enhancer wrote {count} tokens in {elapsed:.1f}s ({count / max(elapsed, 1e-6):.0f} tok/s) "
            f"from a {inputs['input_ids'].shape[1]}-token input"
            + (" (hit cap)" if count >= self.max_new_tokens else ""),
            log_from_all_processes=True,
        )
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        result = parse_enhanced(text, current, user_text)
        if not result.ok:
            log(
                "Prompt enhancer output could not be parsed into TRANSITION/CONTINUATION; using fallback.",
                log_from_all_processes=True,
            )
        return result

    @staticmethod
    def fallback(current: str, user_text: str) -> EnhancedPrompt:
        prompt = _naive_prompt(current, user_text)
        return EnhancedPrompt(transition=prompt, continuation=prompt, raw="", ok=False)
