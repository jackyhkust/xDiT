import torch
import torch._dynamo
import torch.nn.functional as F
from typing import Optional

# Under CFG-parallel the pipeline re-stamps integer sequence-layout attributes
# (_num_pad_tokens / _num_text_tokens) onto this module every __call__, and the
# value changes per call (text length varies from the LLM encoder). torch.compile
# treats nn.Module int attributes as static guards, so without this it recompiles
# the whole forward every iteration (the ~25s step-1 stall seen in cfg+ulysses
# runs). Unspecializing nn.Module ints keeps a single reused graph.
torch._dynamo.config.allow_unspec_int_on_nn_module = True

from xfuser.model_executor.layers.usp import (
    USP,
    attention as usp_attention,
    _ft_c_input_all_to_all,
    _ft_c_output_all_to_all,
)
from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_ulysses_parallel_world_size,
    get_sp_group,
    get_runtime_state,
)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _fp8_attention(query, key, value):
    """Per-tensor FP8 attention using AITER's flash_attn_fp8_pertensor_func.

    Input/output in BHSD layout. Internally converts to BSHD for AITER.
    """
    import aiter
    # BHSD -> BSHD for AITER
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    q_abs_max = query.abs().amax().clamp(min=1e-12)
    k_abs_max = key.abs().amax().clamp(min=1e-12)
    v_abs_max = value.abs().amax().clamp(min=1e-12)
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    q_scale = (q_abs_max / FP8_MAX).float()
    k_scale = (k_abs_max / FP8_MAX).float()
    v_scale = (v_abs_max / FP8_MAX).float()
    q_fp8 = (query / q_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    k_fp8 = (key / k_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    v_fp8 = (value / v_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    out = aiter.flash_attn_fp8_pertensor_func(
        q_fp8, k_fp8, v_fp8,
        q_scale.view(1), k_scale.view(1), v_scale.view(1),
    )
    # BSHD -> BHSD
    return out.transpose(1, 2)


class xFuserIdeogram4AttnProcessor:
    def __init__(self):
        self._use_fp8_attention = False

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        cos, sin = image_rotary_emb
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)

        # (B, L, H, D) -> (B, H, L, D)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self._use_fp8_attention:
            # The AITER FP8 attention kernel is NOT sequence-parallel aware: it
            # attends only over the tokens local to this rank. Under Ulysses SP
            # the sequence is chunked across ranks, so without an all-to-all the
            # rank(s) holding only image tokens never attend to the text tokens
            # (which live on rank 0), and their image region denoises
            # unconditionally -> the output splits into unrelated scenes.
            # Mirror the Ulysses all-to-all that USP() performs: gather the full
            # sequence while scattering heads, run local FP8 attention over the
            # full sequence, then scatter the sequence back.
            if get_ulysses_parallel_world_size() > 1:
                query = _ft_c_input_all_to_all(query)
                key = _ft_c_input_all_to_all(key)
                value = _ft_c_input_all_to_all(value)
                hidden_states = _fp8_attention(query, key, value)
                hidden_states = _ft_c_output_all_to_all(hidden_states)
            else:
                hidden_states = _fp8_attention(query, key, value)
        else:
            hidden_states = USP(query, key, value)

        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3)
        return attn.to_out[0](hidden_states)


def _make_xfuser_ideogram4_transformer_wrapper():
    from diffusers.models.transformers.transformer_ideogram4 import (
        Ideogram4Transformer2DModel,
        Ideogram4Attention,
        OUTPUT_IMAGE_INDICATOR,
        LLM_TOKEN_INDICATOR,
    )
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    class xFuserIdeogram4Transformer2DWrapper(Ideogram4Transformer2DModel):

        def _install_xfuser_processors(self):
            for layer in self.layers:
                layer.attention.set_processor(xFuserIdeogram4AttnProcessor())

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            model = super().from_pretrained(*args, **kwargs)
            model.__class__ = cls
            model._install_xfuser_processors()
            return model

        def _enable_fp8_attention(self):
            for layer in self.layers:
                layer.attention.processor._use_fp8_attention = True

        def _init_sp_state(self):
            try:
                self._sp_rank = get_sequence_parallel_rank()
                self._sp_size = get_sequence_parallel_world_size()
            except AssertionError:
                self._sp_rank = 0
                self._sp_size = 1

        def _set_sequence_layout(self, num_pad_tokens, num_text_tokens, num_image_tokens):
            self._num_pad_tokens = num_pad_tokens
            self._num_text_tokens = num_text_tokens
            self._num_image_tokens = num_image_tokens

        def _chunk_and_pad(self, x, sp_rank, sp_size, pad_amount, dim):
            if pad_amount > 0:
                pad_shape = list(x.shape)
                pad_shape[dim] = pad_amount
                x = torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=dim)
            return torch.chunk(x, sp_size, dim=dim)[sp_rank]

        def _gather_and_unpad(self, x, pad_amount, dim):
            x = get_sp_group().all_gather(x, dim=dim)
            if pad_amount > 0:
                x = x.narrow(dim=dim, start=0, length=x.size(dim) - pad_amount)
            return x

        def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            segment_ids: torch.Tensor,
            indicator: torch.Tensor,
            attention_kwargs: dict | None = None,
            return_dict: bool = True,
        ):
            if not hasattr(self, '_sp_size'):
                self._init_sp_state()
            sp_rank = self._sp_rank
            sp_size = self._sp_size

            if sp_size <= 1:
                return Ideogram4Transformer2DModel.forward(
                    self,
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    indicator=indicator,
                    return_dict=return_dict,
                )

            # SP path: replicate text tokens, chunk only image tokens.
            batch_size, seq_len, in_channels = hidden_states.shape

            if not hasattr(self, '_num_image_tokens') or self._num_image_tokens == 0:
                self._set_sequence_layout(
                    num_pad_tokens=int((indicator[0] == 0).sum()),
                    num_text_tokens=int((indicator[0] == LLM_TOKEN_INDICATOR).sum()),
                    num_image_tokens=int((indicator[0] == OUTPUT_IMAGE_INDICATOR).sum()),
                )
            num_image_tokens = self._num_image_tokens
            num_text_tokens = self._num_text_tokens
            num_pad_tokens = self._num_pad_tokens

            text_start = num_pad_tokens
            image_start = num_pad_tokens + num_text_tokens

            # Pre-process: project text and image, strip padding
            text_enc_raw = encoder_hidden_states[:, text_start:image_start]
            text_enc_raw = self.llm_cond_norm(text_enc_raw)
            text_enc_proj = self.llm_cond_proj(text_enc_raw)

            image_hidden = hidden_states[:, image_start:]
            image_hidden = self.input_proj(image_hidden)

            t_cond = self.t_embedding(timestep)
            if timestep.dim() == 1:
                t_cond = t_cond.unsqueeze(1)
            adaln_input = F.silu(self.adaln_proj(t_cond))

            image_emb = self.embed_image_indicator(
                torch.ones(batch_size, num_image_tokens, dtype=torch.long, device=image_hidden.device)
            )
            text_emb = self.embed_image_indicator(
                torch.zeros(batch_size, num_text_tokens, dtype=torch.long, device=image_hidden.device)
            )

            text_hidden_proj = text_enc_proj + text_emb
            image_hidden_proj = image_hidden + image_emb

            # Build tight [text | image] sequence (no padding)
            full_hidden = torch.cat([text_hidden_proj, image_hidden_proj], dim=1)
            full_len = num_text_tokens + num_image_tokens

            text_pos = position_ids[:, text_start:image_start]
            image_pos = position_ids[:, image_start:]
            full_pos = torch.cat([text_pos, image_pos], dim=1)

            # Chunk the ENTIRE sequence across SP ranks (like FLUX.2)
            img_pad_amount = (sp_size - (full_len % sp_size)) % sp_size
            hidden_states = self._chunk_and_pad(full_hidden, sp_rank, sp_size, img_pad_amount, dim=1)
            pos_ids = self._chunk_and_pad(full_pos, sp_rank, sp_size, img_pad_amount, dim=1)

            cos, sin = self.rotary_emb(pos_ids)
            cos = cos.to(hidden_states.dtype)
            sin = sin.to(hidden_states.dtype)
            image_rotary_emb = (cos, sin)

            # Dummy mask (unused by USP path)
            local_seq_len = hidden_states.shape[1]
            attention_mask = torch.ones(
                batch_size, 1, local_seq_len, local_seq_len,
                dtype=torch.bool, device=hidden_states.device,
            )

            for block in self.layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        block, hidden_states, attention_mask, image_rotary_emb, adaln_input
                    )
                else:
                    hidden_states = block(hidden_states, attention_mask, image_rotary_emb, adaln_input)

            output = self.final_layer(hidden_states, conditioning=adaln_input)

            # Gather full sequence back
            output = self._gather_and_unpad(output, img_pad_amount, dim=1)

            # Reconstruct [pad | text+image]
            pad_output = torch.zeros(
                batch_size, num_pad_tokens, output.shape[-1],
                dtype=output.dtype, device=output.device,
            )
            output = torch.cat([pad_output, output], dim=1)

            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)

    return xFuserIdeogram4Transformer2DWrapper


_wrapper_cls = None

def get_ideogram4_transformer_wrapper_class():
    global _wrapper_cls
    if _wrapper_cls is None:
        _wrapper_cls = _make_xfuser_ideogram4_transformer_wrapper()
    return _wrapper_cls
