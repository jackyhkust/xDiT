import torch
import torch.nn.functional as F
from typing import Optional

from xfuser.model_executor.layers.usp import USP
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

        try:
            sp_size = get_sequence_parallel_world_size()
        except AssertionError:
            sp_size = 1

        if sp_size > 1:
            # (B, L, H, D) -> (B, H, L, D) for USP
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
            try:
                get_runtime_state().increment_step_counter()
                sp_rank = get_sequence_parallel_rank()
                sp_size = get_sequence_parallel_world_size()
            except AssertionError:
                sp_rank = 0
                sp_size = 1

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

            # SP path: strip padding, chunk the entire [text | image]
            # sequence across SP ranks (same approach as FLUX.2).
            batch_size, seq_len, in_channels = hidden_states.shape

            num_image_tokens = (indicator[0] == OUTPUT_IMAGE_INDICATOR).sum().item()
            num_text_tokens = (indicator[0] == LLM_TOKEN_INDICATOR).sum().item()
            num_pad_tokens = seq_len - num_image_tokens - num_text_tokens

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

            # Chunk the ENTIRE sequence across SP ranks
            pad_amount = (sp_size - (full_len % sp_size)) % sp_size
            full_hidden = self._chunk_and_pad(full_hidden, sp_rank, sp_size, pad_amount, dim=1)
            full_pos = self._chunk_and_pad(full_pos, sp_rank, sp_size, pad_amount, dim=1)

            cos, sin = self.rotary_emb(full_pos)
            cos = cos.to(full_hidden.dtype)
            sin = sin.to(full_hidden.dtype)
            image_rotary_emb = (cos, sin)

            # All-ones mask (all valid tokens, padding stripped)
            local_seq_len = full_hidden.shape[1]
            attention_mask = torch.ones(
                batch_size, 1, local_seq_len, local_seq_len,
                dtype=torch.bool, device=full_hidden.device,
            )

            for block in self.layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    full_hidden = self._gradient_checkpointing_func(
                        block, full_hidden, attention_mask, image_rotary_emb, adaln_input
                    )
                else:
                    full_hidden = block(full_hidden, attention_mask, image_rotary_emb, adaln_input)

            output = self.final_layer(full_hidden, conditioning=adaln_input)

            # Gather back to full sequence
            output = self._gather_and_unpad(output, pad_amount, dim=1)

            # Reconstruct [pad | text | image]
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
