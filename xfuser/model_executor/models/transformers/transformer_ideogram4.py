import torch
import torch.nn.functional as F
from typing import Optional

from xfuser.model_executor.layers.usp import USP, attention as usp_attention
from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    get_runtime_state,
)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class xFuserIdeogram4AttnProcessor:
    def __init__(self):
        self._num_text_tokens = 0
        self._sp_size = 1
        self._ulysses_size = 1
        self._ulysses_rank = 0

    def init_sp_state(self):
        try:
            self._sp_size = get_sequence_parallel_world_size()
        except AssertionError:
            self._sp_size = 1
        try:
            from xfuser.core.distributed import (
                get_ulysses_parallel_world_size,
                get_ulysses_parallel_rank,
            )
            self._ulysses_size = get_ulysses_parallel_world_size()
            self._ulysses_rank = get_ulysses_parallel_rank()
        except (AssertionError, ImportError):
            self._ulysses_size = 1
            self._ulysses_rank = 0

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

        sp_size = self._sp_size

        if sp_size > 1 and self._num_text_tokens > 0:
            n_text = self._num_text_tokens
            from xfuser.model_executor.layers.usp import (
                _ft_c_input_all_to_all,
                _ft_c_output_all_to_all,
            )

            # Split text (replicated) and image (chunked) in (B, L, H, D)
            text_q, image_q = query[:, :n_text], query[:, n_text:]
            text_k, image_k = key[:, :n_text], key[:, n_text:]
            text_v, image_v = value[:, :n_text], value[:, n_text:]

            # Transpose to (B, H, L, D)
            text_q = text_q.transpose(1, 2)
            text_k = text_k.transpose(1, 2)
            text_v = text_v.transpose(1, 2)
            image_q = image_q.transpose(1, 2)
            image_k = image_k.transpose(1, 2)
            image_v = image_v.transpose(1, 2)

            # USP all-to-all on image only: split heads, gather sequence
            ulysses_size = self._ulysses_size
            if ulysses_size > 1:
                image_q = _ft_c_input_all_to_all(image_q)
                image_k = _ft_c_input_all_to_all(image_k)
                image_v = _ft_c_input_all_to_all(image_v)

            # Slice text to match the H/P heads on this rank
            ulysses_rank = self._ulysses_rank
            heads_per_rank = text_k.shape[1] // ulysses_size
            text_q_local = text_q[:, heads_per_rank * ulysses_rank : heads_per_rank * (ulysses_rank + 1)].contiguous()
            text_k_local = text_k[:, heads_per_rank * ulysses_rank : heads_per_rank * (ulysses_rank + 1)].contiguous()
            text_v_local = text_v[:, heads_per_rank * ulysses_rank : heads_per_rank * (ulysses_rank + 1)].contiguous()

            # Image attention: image_q x [text_kv + image_kv]
            full_k = torch.cat([text_k_local, image_k], dim=2)
            full_v = torch.cat([text_v_local, image_v], dim=2)
            attn_fn = usp_attention
            image_out = attn_fn(image_q, full_k, full_v)

            # Text attention: text_q x [text_kv + image_kv]
            text_out = attn_fn(text_q_local, full_k, full_v)

            # Reverse USP all-to-all on image: gather heads, split sequence
            if ulysses_size > 1:
                image_out = _ft_c_output_all_to_all(image_out)

            # Gather text heads back across SP ranks
            if ulysses_size > 1:
                text_out = get_sp_group().all_gather(text_out.contiguous(), dim=1)

            hidden_states = torch.cat([text_out.transpose(1, 2), image_out.transpose(1, 2)], dim=1)

        elif sp_size > 1:
            # Image-only path (unconditional transformer)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            hidden_states = USP(query, key, value)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            from diffusers.models.attention_dispatch import dispatch_attention_fn
            hidden_states = dispatch_attention_fn(
                query, key, value, attn_mask=attention_mask,
            )

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

        def _init_sp_state(self):
            try:
                self._sp_rank = get_sequence_parallel_rank()
                self._sp_size = get_sequence_parallel_world_size()
            except AssertionError:
                self._sp_rank = 0
                self._sp_size = 1
            for layer in self.layers:
                layer.attention.processor.init_sp_state()

        def _set_processor_text_tokens(self, num_text_tokens):
            for layer in self.layers:
                layer.attention.processor._num_text_tokens = num_text_tokens

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

            # Chunk ONLY image tokens (always power-of-2 for standard resolutions)
            img_pad_amount = (sp_size - (num_image_tokens % sp_size)) % sp_size
            image_hidden_proj = self._chunk_and_pad(image_hidden_proj, sp_rank, sp_size, img_pad_amount, dim=1)

            # Position IDs: replicate text, chunk image
            text_pos = position_ids[:, text_start:image_start]
            image_pos = position_ids[:, image_start:]
            image_pos = self._chunk_and_pad(image_pos, sp_rank, sp_size, img_pad_amount, dim=1)

            # Concatenate: [text_replicated | image_chunk]
            hidden_states = torch.cat([text_hidden_proj, image_hidden_proj], dim=1)
            pos_ids = torch.cat([text_pos, image_pos], dim=1)

            cos, sin = self.rotary_emb(pos_ids)
            cos = cos.to(hidden_states.dtype)
            sin = sin.to(hidden_states.dtype)
            image_rotary_emb = (cos, sin)

            # Tell attn processor where text ends
            self._set_processor_text_tokens(num_text_tokens)

            # Dummy mask (unused by USP path, used by non-SP fallback)
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

            self._set_processor_text_tokens(0)

            # Split text and image output, gather image
            text_output = output[:, :num_text_tokens]
            image_output = output[:, num_text_tokens:]
            image_output = self._gather_and_unpad(image_output, img_pad_amount, dim=1)

            # Reconstruct [pad | text | image]
            pad_output = torch.zeros(
                batch_size, num_pad_tokens, output.shape[-1],
                dtype=output.dtype, device=output.device,
            )
            output = torch.cat([pad_output, text_output, image_output], dim=1)

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
