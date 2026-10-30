import functools
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch._dynamo import mark_static_address
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux3 import (
    Flux3Attention,
    Flux3ModeTransformerBlock,
    Flux3SingleTransformerBlock,
    Flux3Transformer2DModel,
    _get_qkv_projections,
)

from xfuser.core.distributed import (
    get_runtime_state,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from xfuser.core.distributed.fp8_comms import register_fp8_comms_eligible_modules
from xfuser.model_executor.layers.usp import USP, attention
from xfuser.model_executor.models.transformers.transformers_utils import (
    chunk_and_pad_sequence,
    gather_and_unpad,
)


class xFuserFlux3AttnProcessor:
    """Flux3AttnProcessor with the attention call routed through USP.

    Each sequence stage (text, image and reference-image modes, then the joint
    blocks) shares one instance. Before a stage runs, the transformer sets
    ``valid_kv_len`` to the stage's real length when the SP pad made the sequence
    longer, so the pad keys are dropped after the all-to-all, and ``local`` when
    every rank holds the whole stream, so attention runs without collectives.
    """

    def __init__(self):
        self.valid_kv_len: Optional[int] = None
        self.local = False

    def __call__(
        self,
        attn: Flux3Attention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError("FLUX.3 attention takes no mask.")
        query, key, value = _get_qkv_projections(attn, hidden_states)
        return self.attend(attn, query, key, value, image_rotary_emb)

    def attend(
        self,
        attn: Flux3Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Attention and output projection from already projected q, k and v."""
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # QK norm and RoPE in fp32 with one cast back, as upstream.
        query = attn.norm_q(query.float())
        key = attn.norm_k(key.float())

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        query, key = query.to(value.dtype), key.to(value.dtype)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.local:
            hidden_states = attention(query, key, value)
        else:
            attention_kwargs = None if self.valid_kv_len is None else {"valid_kv_len": self.valid_kv_len}
            hidden_states = USP(
                query, key, value, attn_layer=attn, attention_kwargs=attention_kwargs, combine_qkv_a2a=True
            )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def flux3_attn_modules(transformer) -> list[torch.nn.Module]:
    """Return every FLUX.3 attention module that executes USP, in execution order."""
    blocks = (
        *transformer.txt_mode_blocks,
        *transformer.mode_blocks["image"],
        *transformer.mode_blocks["image_cond"],
        *transformer.single_transformer_blocks,
    )
    return [block.attn for block in blocks]


def _fused_attention_and_ff(block, norm_hidden_states, image_rotary_emb):
    """The block's attention and feed-forward outputs from one shared input GEMM."""
    query, key, value, ff_hidden = block.fused_in(norm_hidden_states).split(block.fused_in_split, dim=-1)
    attn_output = block.attn.processor.attend(block.attn, query, key, value, image_rotary_emb)
    ff_output = block.ff.linear_out(block.ff.act_fn(ff_hidden))
    return attn_output, ff_output


class _FusedFlux3ModeTransformerBlock(Flux3ModeTransformerBlock):
    """Flux3ModeTransformerBlock with q, k, v and the feed-forward input in one GEMM."""

    def forward(self, hidden_states, temb_mod, image_rotary_emb):
        mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        attn_output, ff_output = _fused_attention_and_ff(self, norm_hidden_states, image_rotary_emb)
        return hidden_states + mod_gate * (attn_output + ff_output)


class _FusedFlux3SingleTransformerBlock(Flux3SingleTransformerBlock):
    """Flux3SingleTransformerBlock with q, k, v and the feed-forward input in one GEMM."""

    def forward(self, hidden_states, encoder_hidden_states, temb_mod, temb_mod_context, image_rotary_emb):
        mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)
        c_mod_shift, c_mod_scale, c_mod_gate = torch.chunk(temb_mod_context, 3, dim=-1)

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        norm_encoder_hidden_states = self.norm(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_mod_scale) * norm_encoder_hidden_states + c_mod_shift

        combined = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        attn_output, ff_output = _fused_attention_and_ff(self, combined, image_rotary_emb)
        output = attn_output + ff_output

        context_output, content_output = torch.split(
            output, [encoder_hidden_states.shape[1], hidden_states.shape[1]], dim=1
        )
        encoder_hidden_states = encoder_hidden_states + c_mod_gate * context_output
        hidden_states = hidden_states + mod_gate * content_output
        return encoder_hidden_states, hidden_states


_FUSED_BLOCK_CLASSES = {
    Flux3ModeTransformerBlock: _FusedFlux3ModeTransformerBlock,
    Flux3SingleTransformerBlock: _FusedFlux3SingleTransformerBlock,
}


def _concatenated_linear(layers) -> Optional[torch.nn.Module]:
    """One layer equal to the given layers' outputs concatenated, or None if their kind
    cannot be stacked (FP8 tensor subclasses, the hybrid FP4 layer, bias)."""
    from xfuser.model_executor.layers.mxfp4_linear import xFuserMXFP4Linear

    if all(type(layer) is xFuserMXFP4Linear for layer in layers):
        return xFuserMXFP4Linear.concatenated(layers)
    plain = all(
        type(layer) is torch.nn.Linear and layer.bias is None and type(layer.weight.data) is torch.Tensor
        for layer in layers
    )
    if not plain:
        return None
    fused = torch.nn.Linear(
        layers[0].in_features,
        sum(layer.out_features for layer in layers),
        bias=False,
        device="meta",
    )
    fused.weight = torch.nn.Parameter(torch.cat([layer.weight.data for layer in layers]), requires_grad=False)
    return fused


def fuse_block_input_projections(transformer) -> int:
    """Give every block whose q, k, v and feed-forward input layers can be stacked one
    GEMM for all four, and return how many blocks changed.

    The four read the same modulated input, so this also quantizes that input once
    instead of four times under MXFP4.
    """
    fused_blocks = 0
    for block in (
        *transformer.txt_mode_blocks,
        *transformer.mode_blocks["image"],
        *transformer.mode_blocks["image_cond"],
        *transformer.single_transformer_blocks,
    ):
        fused_class = _FUSED_BLOCK_CLASSES.get(type(block))
        if fused_class is None:
            continue
        layers = (block.attn.to_q, block.attn.to_k, block.attn.to_v, block.ff.linear_in)
        fused = _concatenated_linear(layers)
        if fused is None:
            continue
        block.fused_in = fused
        block.fused_in_split = [layer.out_features for layer in layers]
        del block.attn.to_q, block.attn.to_k, block.attn.to_v, block.ff.linear_in
        block.__class__ = fused_class
        fused_blocks += 1
    return fused_blocks


class xFuserFlux3Transformer2DWrapper(Flux3Transformer2DModel):
    """Sequence-parallel FLUX.3 transformer.

    Every sequence is padded at its end to a multiple of the SP degree and split
    evenly across ranks, so after the Ulysses all-to-all the pad is a uniform
    suffix that ``valid_kv_len`` removes:

    - Per-mode blocks: the text, image and reference-image streams are attended
      separately. The image and reference streams are sharded, then gathered. A
      text stream of at most ``replicate_text_max_tokens`` runs whole on every
      rank instead, which skips its collectives for a little repeated compute.
    - Joint blocks: ``[text, image, reference]`` is sharded as one sequence. A
      rank's chunk can straddle the text/content boundary, so each block gets
      the chunk's text part (possibly empty) as ``encoder_hidden_states`` and the
      rest as ``hidden_states``. The blocks run unmodified; with one rank this is
      exactly the upstream call.
    """

    # Measured on 8x MI350X at 1024² (u8): replicating 80 or 400 text tokens saves about
    # 1.5 ms per eager forward of 38 ms; under HIP graphs the difference vanishes.
    replicate_text_max_tokens = 512

    # ConfigMixin picks the config keys to pass from this signature. Wrapping upstream's
    # __init__ shows it upstream's parameters; behind a bare **kwargs every key, flow_map
    # included, would fall back to its default.
    @functools.wraps(Flux3Transformer2DModel.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._txt_processor = xFuserFlux3AttnProcessor()
        self._image_processor = xFuserFlux3AttnProcessor()
        self._image_cond_processor = xFuserFlux3AttnProcessor()
        self._joint_processor = xFuserFlux3AttnProcessor()
        stages = (
            (self.txt_mode_blocks, self._txt_processor),
            (self.mode_blocks["image"], self._image_processor),
            (self.mode_blocks["image_cond"], self._image_cond_processor),
            (self.single_transformer_blocks, self._joint_processor),
        )
        for blocks, processor in stages:
            for block in blocks:
                block.attn.processor = processor
        register_fp8_comms_eligible_modules(self, flux3_attn_modules(self))
        self._block_input_buffers: Dict[tuple, torch.Tensor] = {}
        self._step_constants: Dict[str, tuple] = {}
        # One transformer call per scheduled step: the hybrid GEMM and attention schedules
        # pick the step's precision before the blocks run.
        self.register_forward_pre_hook(lambda module, args: get_runtime_state().increment_step_counter())

    def _block_input(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """``tensor`` in a persistent buffer that every block call of a step reads.

        The compiled blocks replay as CUDA/HIP graphs, which copy each input of
        unknown address into their own memory before every replay: 28 joint blocks
        per step would copy the same RoPE tables and modulations 28 times. A buffer
        per name and shape, marked static, keeps one address, so it is filled once
        per step instead. Stream order keeps a later fill behind earlier readers.
        """
        key = (name, tuple(tensor.shape), tensor.dtype, tensor.device)
        buffer = self._block_input_buffers.get(key)
        if buffer is None:
            buffer = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            mark_static_address(buffer)
            self._block_input_buffers[key] = buffer
        buffer.copy_(tensor)
        return buffer

    def _step_constant(self, name: str, source: torch.Tensor, compute) -> Any:
        """``compute(source)``, reused while the steps pass the same, unmodified ``source``.

        The denoise loop hands every step of a call the same ids and prompt embeddings, so
        their RoPE tables (built in float64 from many small kernels) and text projection need
        computing once per call. The entry holds ``source``, so a later tensor cannot
        reuse its identity; in-place writes bump its version. Inference tensors have no
        version counter, so for them identity alone decides: the pipeline never writes its
        conditioning in place. Autograd always recomputes.
        """
        if torch.is_grad_enabled():
            return compute(source)
        version = None if source.is_inference() else source._version
        cached = self._step_constants.get(name)
        if cached is not None and cached[0] is source and cached[1] == version:
            return cached[2]
        result = compute(source)
        self._step_constants[name] = (source, version, result)
        return result

    def _modulation(self, name: str, mod_row: torch.Tensor, seq_len: int) -> torch.Tensor:
        """The (batch, seq_len, 3 * dim) modulation of a mode whose tokens share one row."""
        mod = self._block_input(name, mod_row).expand(-1, seq_len, -1)
        mark_static_address(mod)
        return mod

    def _content_modulation(self, mod_rows, stream_lens, start: int, length: int) -> torch.Tensor:
        """The joint-block modulation of this rank's content tokens: positions ``start`` to
        ``start + length`` of the concatenated content streams, each with its own row.
        SP pad positions past the end take the last stream's row; their outputs are dropped."""
        pieces = []
        offset = 0
        for index, (mod_row, stream_len) in enumerate(zip(mod_rows, stream_lens)):
            end = offset + stream_len if index < len(mod_rows) - 1 else start + length
            overlap = min(end, start + length) - max(offset, start)
            if overlap > 0:
                pieces.append((index, mod_row, overlap))
            offset += stream_len
        if len(pieces) <= 1:
            index, mod_row, _ = pieces[0] if pieces else (0, mod_rows[0], 0)
            return self._modulation(f"joint.content_mod{index}", mod_row, length)
        return torch.cat([mod_row.expand(-1, overlap, -1) for _, mod_row, overlap in pieces], dim=1)

    def _rotary(self, name: str, rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._block_input(f"{name}.cos", rotary_emb[0]), self._block_input(f"{name}.sin", rotary_emb[1])

    def _run_mode_blocks(
        self,
        name: str,
        blocks: torch.nn.ModuleList,
        processor: xFuserFlux3AttnProcessor,
        hidden_states: torch.Tensor,
        mod_row: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        local: bool = False,
    ) -> torch.Tensor:
        processor.local = local
        sp_world_size = get_sequence_parallel_world_size()
        seq_len = hidden_states.shape[1]
        pad = 0 if local else (sp_world_size - seq_len % sp_world_size) % sp_world_size
        processor.valid_kv_len = seq_len if pad else None
        if not local:
            sp_rank = get_sequence_parallel_rank()
            hidden_states = chunk_and_pad_sequence(hidden_states, sp_rank, sp_world_size, pad, dim=1)
            rotary_emb = tuple(chunk_and_pad_sequence(r, sp_rank, sp_world_size, pad, dim=0) for r in rotary_emb)

        temb_mod = self._modulation(f"{name}.mod", mod_row, hidden_states.shape[1])
        rotary_emb = self._rotary(name, rotary_emb)
        for block in blocks:
            hidden_states = block(hidden_states, temb_mod, rotary_emb)
        return hidden_states if local else gather_and_unpad(hidden_states, pad, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        image_cond_hidden_states: Optional[torch.Tensor] = None,
        image_cond_ids: Optional[torch.Tensor] = None,
        target_timestep: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        if image_cond_hidden_states is not None and image_cond_ids is None:
            raise ValueError("`image_cond_hidden_states` requires `image_cond_ids`.")
        if image_cond_hidden_states is None and image_cond_ids is not None:
            raise ValueError("`image_cond_ids` was passed without `image_cond_hidden_states`.")
        if (target_timestep is not None) != bool(self.config.flow_map):
            raise ValueError("`target_timestep` is required by, and only by, a flow-map transformer (`flow_map=True`).")

        batch_size, image_seq_len = hidden_states.shape[:2]
        dtype = hidden_states.dtype

        # 1-4. Timestep embeddings, modulations, input projections and RoPE, as upstream.
        image_temb = self.time_proj(timestep.to(dtype) * 1000)
        image_temb = self.timestep_embedder(image_temb.to(dtype))[:, None, :]
        zero_temb = self.time_proj(torch.zeros(batch_size, device=hidden_states.device, dtype=dtype))
        zero_temb = self.timestep_embedder(zero_temb.to(dtype))[:, None, :]
        if target_timestep is not None:
            target_temb = self.time_proj(target_timestep.to(dtype) * 1000)
            target_temb = self.target_timestep_embedder(target_temb.to(dtype))[:, None, :]
            image_temb = image_temb + target_temb
            zero_temb = zero_temb + target_temb
        zero_temb_act = F.silu(zero_temb)

        # Every token of a mode has the same timestep, so each modulation is computed on
        # one row and broadcast instead of on every token of every rank.
        txt_early_mod = self.early_stream_modulations["txt"](zero_temb_act)
        txt_single_mod = self.single_stream_modulations["txt"](zero_temb_act)
        encoder_hidden_states = self._step_constant("context", encoder_hidden_states, self.context_embedder)
        txt_rotary_emb = self._step_constant("txt_ids", txt_ids, self.pos_embed)

        image_temb_act = F.silu(image_temb)
        image_early_mod = self.early_stream_modulations["image"](image_temb_act)
        image_single_mod = self.single_stream_modulations["image"](image_temb_act)
        hidden_states = self.x_embedders["image"](hidden_states)
        image_rotary_emb = self._step_constant("img_ids", img_ids, self.pos_embed)

        if image_cond_hidden_states is not None:
            cond_early_mod = self.early_stream_modulations["image_cond"](zero_temb_act)
            cond_single_mod = self.single_stream_modulations["image_cond"](zero_temb_act)
            image_cond_hidden_states = self.x_embedders["image_cond"](image_cond_hidden_states)
            cond_rotary_emb = self._step_constant("image_cond_ids", image_cond_ids, self.pos_embed)

        # 5. Per-mode blocks, each mode attended on its own; short text runs whole on every rank.
        replicate_text = (
            get_sequence_parallel_world_size() > 1 and encoder_hidden_states.shape[1] <= self.replicate_text_max_tokens
        )
        encoder_hidden_states = self._run_mode_blocks(
            "txt",
            self.txt_mode_blocks,
            self._txt_processor,
            encoder_hidden_states,
            txt_early_mod,
            txt_rotary_emb,
            local=replicate_text,
        )
        hidden_states = self._run_mode_blocks(
            "image", self.mode_blocks["image"], self._image_processor, hidden_states, image_early_mod, image_rotary_emb
        )
        streams = [
            (encoder_hidden_states, txt_single_mod, txt_rotary_emb),
            (hidden_states, image_single_mod, image_rotary_emb),
        ]
        if image_cond_hidden_states is not None:
            image_cond_hidden_states = self._run_mode_blocks(
                "image_cond",
                self.mode_blocks["image_cond"],
                self._image_cond_processor,
                image_cond_hidden_states,
                cond_early_mod,
                cond_rotary_emb,
            )
            streams.append((image_cond_hidden_states, cond_single_mod, cond_rotary_emb))

        # 6. Joint blocks over [text, image, reference], sharded as one sequence.
        txt_seq_len = encoder_hidden_states.shape[1]
        joint_states = torch.cat([s[0] for s in streams], dim=1)
        joint_rotary_emb = (
            torch.cat([s[2][0] for s in streams], dim=0),
            torch.cat([s[2][1] for s in streams], dim=0),
        )
        joint_seq_len = joint_states.shape[1]

        sp_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()
        pad = (sp_world_size - joint_seq_len % sp_world_size) % sp_world_size
        self._joint_processor.valid_kv_len = joint_seq_len if pad else None
        joint_states = chunk_and_pad_sequence(joint_states, sp_rank, sp_world_size, pad, dim=1)
        joint_rotary_emb = self._rotary(
            "joint", tuple(chunk_and_pad_sequence(r, sp_rank, sp_world_size, pad, dim=0) for r in joint_rotary_emb)
        )

        # Text tokens of this rank's chunk: the leading part, possibly none.
        chunk_len = joint_states.shape[1]
        local_txt_len = min(max(txt_seq_len - sp_rank * chunk_len, 0), chunk_len)
        context_states = joint_states[:, :local_txt_len]
        content_states = joint_states[:, local_txt_len:]
        context_mod = self._modulation("joint.txt_mod", txt_single_mod, local_txt_len)
        content_mod = self._content_modulation(
            [s[1] for s in streams[1:]],
            [s[0].shape[1] for s in streams[1:]],
            start=max(sp_rank * chunk_len - txt_seq_len, 0),
            length=chunk_len - local_txt_len,
        )

        for block in self.single_transformer_blocks:
            context_states, content_states = block(
                hidden_states=content_states,
                encoder_hidden_states=context_states,
                temb_mod=content_mod,
                temb_mod_context=context_mod,
                image_rotary_emb=joint_rotary_emb,
            )

        # 7. Output layers on this rank's image tokens, which follow its text tokens. They
        #    are per token, so only the output channels need gathering. Every image token
        #    has the same modulation, so the one image_temb row broadcasts.
        local_image_len = min(max(txt_seq_len + image_seq_len - sp_rank * chunk_len, 0), chunk_len) - local_txt_len
        hidden_states = content_states[:, :local_image_len]
        hidden_states = self.norm_out(hidden_states, image_temb)
        output = self.proj_out(hidden_states)
        output = F.pad(output, (0, 0, local_txt_len, chunk_len - local_txt_len - local_image_len))
        output = gather_and_unpad(output, pad, dim=1)[:, txt_seq_len : txt_seq_len + image_seq_len]

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
