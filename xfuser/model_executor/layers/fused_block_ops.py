"""Fused FLUX.3 block kernels around the GEMMs: what Inductor cannot fuse across a custom
op, a collective or a separately compiled block.

- ``qk_norm_rope_seq_major_send``: the q/k RMSNorm and RoPE of a block's projected q, k, v,
  written straight into the Ulysses sequence-major send buffer ``[P, s, b, 3, h/P, d]``
  (see ``usp._seq_major_qkv_send``) with v copied alongside, instead of a norm kernel and a
  pack kernel.
- ``qk_norm_rope``: the same q and k as one ``[b, s, 2, h, d]`` buffer, for the paths that
  exchange q, k and v some other way, or not at all.
- ``gated_residual_layernorm_modulate`` and its ``_mxfp4_quant`` variant: the previous
  block's gated residual ``x + gate * (attn + ff)`` of one or two segments (or
  ``x + gate * out`` after a merged output GEMM), then
  ``(1 + scale) * layer_norm(.) + shift`` of the result, the next block's input, in bf16
  or quantized as ``mxfp4_fused_quant.layernorm_modulate_mxfp4_quant`` quantizes it.

The math follows upstream: q/k norm and RoPE in fp32 with one rounding to the output
dtype; the residual rounds once to the hidden-state dtype, which the LayerNorm then reads,
and the modulated norm rounds once more.
"""

from typing import List, Tuple

import torch

from xfuser.model_executor.layers import mxfp4_fused_quant

try:
    import triton
    import triton.language as tl

    FUSED_BLOCK_OPS_AVAILABLE = True
except ImportError:
    FUSED_BLOCK_OPS_AVAILABLE = False

# Heads per program and warps of the q/k kernel, and warps of the row kernels: the fastest
# at FLUX.3's shapes on MI355X (24 heads of 128, rows of 3072).
_QK_HEADS_PER_PROGRAM = 8
_QK_NUM_WARPS = 2
_ROW_NUM_WARPS = 4


if FUSED_BLOCK_OPS_AVAILABLE:

    @triton.jit
    def _rms_norm_rope(x, weight, cos, sin, eps, HEADS: tl.constexpr, HEAD_DIM: tl.constexpr):
        """RMSNorm over the last dimension of fp32 ``x`` ``[HEADS, HEAD_DIM]``, then RoPE on
        interleaved pairs: ``(x0, x1)`` rotates by ``(-x1, x0)``."""
        rstd = tl.rsqrt(tl.sum(x * x, axis=1) / HEAD_DIM + eps)
        y = x * rstd[:, None] * weight
        even, odd = tl.split(tl.reshape(y, [HEADS, HEAD_DIM // 2, 2]))
        rotated = tl.reshape(tl.join(-odd, even), [HEADS, HEAD_DIM])
        return y * cos + rotated * sin

    @triton.jit
    def _qk_norm_rope_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        seq_len,
        heads,
        heads_per_rank,
        eps,
        in_stride_b,
        in_stride_s,
        rope_stride_s,
        out_stride_rank,
        out_stride_s,
        out_stride_b,
        out_stride_which,
        WRITE_V: tl.constexpr,
        HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        index = tl.program_id(0)
        b = index // seq_len
        s = index % seq_len
        head = tl.program_id(1) * HEADS + tl.arange(0, HEADS)[:, None]
        dim = tl.arange(0, HEAD_DIM)[None, :]
        mask = head < heads
        in_offsets = b.to(tl.int64) * in_stride_b + s.to(tl.int64) * in_stride_s + head * HEAD_DIM + dim
        out_offsets = (
            (head // heads_per_rank).to(tl.int64) * out_stride_rank
            + s.to(tl.int64) * out_stride_s
            + b.to(tl.int64) * out_stride_b
            + (head % heads_per_rank) * HEAD_DIM
            + dim
        )
        out_dtype = out_ptr.dtype.element_ty
        cos = tl.load(cos_ptr + s.to(tl.int64) * rope_stride_s + dim).to(tl.float32)
        sin = tl.load(sin_ptr + s.to(tl.int64) * rope_stride_s + dim).to(tl.float32)

        q = tl.load(q_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
        q = _rms_norm_rope(q, tl.load(q_weight_ptr + dim).to(tl.float32), cos, sin, eps, HEADS, HEAD_DIM)
        tl.store(out_ptr + out_offsets, q.to(out_dtype), mask=mask)
        k = tl.load(k_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
        k = _rms_norm_rope(k, tl.load(k_weight_ptr + dim).to(tl.float32), cos, sin, eps, HEADS, HEAD_DIM)
        tl.store(out_ptr + out_stride_which + out_offsets, k.to(out_dtype), mask=mask)
        if WRITE_V:
            v = tl.load(v_ptr + in_offsets, mask=mask, other=0.0)
            tl.store(out_ptr + 2 * out_stride_which + out_offsets, v.to(out_dtype), mask=mask)

    @triton.jit
    def _segment_row(ptr, stride_b, stride_s, b, s):
        return ptr + b.to(tl.int64) * stride_b + s.to(tl.int64) * stride_s

    @triton.jit
    def _gated_residual_layernorm_modulate_kernel(
        x0_ptr,
        gate0_ptr,
        shift0_ptr,
        mod_scale0_ptr,
        x1_ptr,
        gate1_ptr,
        shift1_ptr,
        mod_scale1_ptr,
        branch0_ptr,
        branch1_ptr,
        hidden_ptr,
        out_ptr,
        fp4_ptr,
        scale_ptr,
        seq_len0,
        rows_per_batch,
        cols,
        eps,
        x0_stride_b,
        x0_stride_s,
        gate0_stride_b,
        gate0_stride_s,
        shift0_stride_b,
        shift0_stride_s,
        mod_scale0_stride_b,
        mod_scale0_stride_s,
        x1_stride_b,
        x1_stride_s,
        gate1_stride_b,
        gate1_stride_s,
        shift1_stride_b,
        shift1_stride_s,
        mod_scale1_stride_b,
        mod_scale1_stride_s,
        branch0_stride_b,
        branch0_stride_s,
        branch1_stride_b,
        branch1_stride_s,
        scale_cols,
        TWO_BRANCHES: tl.constexpr,
        QUANTIZE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One program per row of the joint ``[b, rows_per_batch]`` sequence: its first
        ``seq_len0`` rows come from segment 0, the rest from segment 1."""
        row = tl.program_id(0)
        b = row // rows_per_batch
        joint_s = row % rows_per_batch
        if joint_s < seq_len0:
            x_row = _segment_row(x0_ptr, x0_stride_b, x0_stride_s, b, joint_s)
            gate_row = _segment_row(gate0_ptr, gate0_stride_b, gate0_stride_s, b, joint_s)
            shift_row = _segment_row(shift0_ptr, shift0_stride_b, shift0_stride_s, b, joint_s)
            mod_scale_row = _segment_row(mod_scale0_ptr, mod_scale0_stride_b, mod_scale0_stride_s, b, joint_s)
        else:
            s = joint_s - seq_len0
            x_row = _segment_row(x1_ptr, x1_stride_b, x1_stride_s, b, s)
            gate_row = _segment_row(gate1_ptr, gate1_stride_b, gate1_stride_s, b, s)
            shift_row = _segment_row(shift1_ptr, shift1_stride_b, shift1_stride_s, b, s)
            mod_scale_row = _segment_row(mod_scale1_ptr, mod_scale1_stride_b, mod_scale1_stride_s, b, s)
        col = tl.arange(0, BLOCK)
        mask = col < cols

        x = tl.load(x_row + col, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_row + col, mask=mask, other=0.0).to(tl.float32)
        branch0_row = _segment_row(branch0_ptr, branch0_stride_b, branch0_stride_s, b, joint_s)
        branch = tl.load(branch0_row + col, mask=mask, other=0.0).to(tl.float32)
        if TWO_BRANCHES:
            branch1_row = _segment_row(branch1_ptr, branch1_stride_b, branch1_stride_s, b, joint_s)
            branch += tl.load(branch1_row + col, mask=mask, other=0.0).to(tl.float32)
        hidden = (x + gate * branch).to(hidden_ptr.dtype.element_ty)
        tl.store(hidden_ptr + row.to(tl.int64) * cols + col, hidden, mask=mask)

        hidden = hidden.to(tl.float32)
        mean = tl.sum(hidden, axis=0) / cols
        centered = tl.where(mask, hidden - mean, 0.0)
        rstd = tl.rsqrt(tl.sum(centered * centered, axis=0) / cols + eps)
        shift = tl.load(shift_row + col, mask=mask, other=0.0).to(tl.float32)
        mod_scale = tl.load(mod_scale_row + col, mask=mask, other=0.0).to(tl.float32)
        y = (1.0 + mod_scale) * (centered * rstd) + shift
        if QUANTIZE:
            y = tl.where(mask, y.to(tl.bfloat16).to(tl.float32), 0.0)
            mxfp4_fused_quant._store_mxfp4_row(y, row, 0, cols, fp4_ptr, scale_ptr, scale_cols, BLOCK)
        else:
            tl.store(out_ptr + row.to(tl.int64) * cols + col, y.to(out_ptr.dtype.element_ty), mask=mask)


def _check_qk_inputs(query, key, value, q_weight, k_weight, cos, sin, heads):
    head_dim = query.shape[-1] // heads
    if head_dim * heads != query.shape[-1] or head_dim & (head_dim - 1) or head_dim < 16:
        raise ValueError(f"{query.shape[-1]} columns are not {heads} heads of a power of 2 of at least 16.")
    for name, tensor in (("key", key), ("value", value)):
        if tensor.shape != query.shape or tensor.stride() != query.stride():
            raise ValueError(f"{name} must match query's shape and strides, got {tensor.shape}, {tensor.stride()}.")
    if query.stride(-1) != 1:
        raise ValueError(f"q, k and v must be contiguous in their last dimension, got strides {query.stride()}.")
    for name, tensor in (("q_weight", q_weight), ("k_weight", k_weight)):
        if tensor.shape != (head_dim,) or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous [{head_dim}] tensor, got {tuple(tensor.shape)}.")
    if cos.shape != (query.shape[1], head_dim) or cos.stride(-1) != 1 or sin.stride() != cos.stride():
        raise ValueError(f"cos and sin must be [{query.shape[1]}, {head_dim}] rows, got {tuple(cos.shape)}.")
    return head_dim


def _launch_qk_norm_rope(query, key, value, q_weight, k_weight, eps, cos, sin, heads, out, seq_major):
    """Run the q/k kernel on ``[b, s, h * d]`` q, k, v into ``out``: the send buffer
    ``[P, s, b, 3, h/P, d]`` if ``seq_major``, else ``[b, s, 2, h, d]`` q and k."""
    batch, seq_len, _ = query.shape
    head_dim = query.shape[-1] // heads
    if seq_major:
        heads_per_rank = out.shape[4]
        strides = (out.stride(0), out.stride(1), out.stride(2), out.stride(3))
    else:
        heads_per_rank = heads
        strides = (0, out.stride(1), out.stride(0), out.stride(2))
    if batch * seq_len:
        _qk_norm_rope_kernel[(batch * seq_len, triton.cdiv(heads, _QK_HEADS_PER_PROGRAM))](
            query,
            key,
            value,
            q_weight,
            k_weight,
            cos,
            sin,
            out,
            seq_len,
            heads,
            heads_per_rank,
            eps,
            query.stride(0),
            query.stride(1),
            cos.stride(0),
            *strides,
            WRITE_V=seq_major,
            HEADS=_QK_HEADS_PER_PROGRAM,
            HEAD_DIM=head_dim,
            num_warps=_QK_NUM_WARPS,
        )
    return out


@torch.library.custom_op("xfuser::qk_norm_rope_seq_major_send", mutates_args=())
def qk_norm_rope_seq_major_send(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    world_size: int,
) -> torch.Tensor:
    """The ``[P, s, b, 3, h/P, d]`` Ulysses send buffer of ``[b, s, h * d]`` q, k, v (column
    slices of one GEMM output, so sharing strides): q and k RMSNormed by their weights and
    rotated by the ``[s, d]`` cos/sin rows, v as is."""
    head_dim = _check_qk_inputs(query, key, value, q_weight, k_weight, cos, sin, heads)
    if heads % world_size:
        raise ValueError(f"{heads} heads do not split over {world_size} ranks.")
    batch, seq_len, _ = query.shape
    out = query.new_empty(world_size, seq_len, batch, 3, heads // world_size, head_dim)
    return _launch_qk_norm_rope(query, key, value, q_weight, k_weight, eps, cos, sin, heads, out, seq_major=True)


@qk_norm_rope_seq_major_send.register_fake
def _(query, key, value, q_weight, k_weight, eps, cos, sin, heads, world_size):
    batch, seq_len, cols = query.shape
    return query.new_empty(world_size, seq_len, batch, 3, heads // world_size, cols // heads)


@torch.library.custom_op("xfuser::qk_norm_rope", mutates_args=())
def qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
) -> torch.Tensor:
    """The RMSNormed, rotated ``[b, s, h * d]`` q and k as one ``[b, s, 2, h, d]`` buffer."""
    head_dim = _check_qk_inputs(query, key, key, q_weight, k_weight, cos, sin, heads)
    batch, seq_len, _ = query.shape
    out = query.new_empty(batch, seq_len, 2, heads, head_dim)
    return _launch_qk_norm_rope(query, key, key, q_weight, k_weight, eps, cos, sin, heads, out, seq_major=False)


@qk_norm_rope.register_fake
def _(query, key, q_weight, k_weight, eps, cos, sin, heads):
    batch, seq_len, cols = query.shape
    return query.new_empty(batch, seq_len, 2, heads, cols // heads)


def _gated_residual_layernorm_modulate(xs, gates, shifts, scales, branches, eps, quantize):
    """The joint ``[b, sum s_i, k]`` hidden state, then the bf16 modulated norm or its MXFP4
    values and scales, of one or two segments ``xs[i]`` ``[b, s_i, k]`` laid end to end along
    the sequence, whose branch output is the sum of the matching rows of the one or two
    ``branches`` (a block's attention and feed-forward outputs, or their merged GEMM's)."""
    if not 1 <= len(xs) <= 2 or not len(xs) == len(gates) == len(shifts) == len(scales):
        raise ValueError(f"Expected one or two segments with a gate, shift and scale each, got {len(xs)} segments.")
    if not 1 <= len(branches) <= 2:
        raise ValueError(f"Expected one or two branch outputs, got {len(branches)}.")
    batch, cols = xs[0].shape[0], xs[0].shape[-1]
    rows_per_batch = sum(x.shape[1] for x in xs)
    if any(branch.shape != (batch, rows_per_batch, cols) for branch in branches):
        shapes = [tuple(branch.shape) for branch in branches]
        raise ValueError(f"The branch outputs must be [{batch}, {rows_per_batch}, {cols}], got {shapes}.")
    hidden = xs[0].new_empty(batch, rows_per_batch, cols)
    if quantize:
        out = hidden.new_empty(0)
        fp4, quant_scale = mxfp4_fused_quant.mxfp4_buffers(batch * rows_per_batch, cols, hidden.device)
    else:
        out = torch.empty_like(hidden)
        fp4 = quant_scale = hidden.new_empty(0, dtype=torch.uint8)
    segments = []
    for x, gate, shift, scale in zip(xs, gates, shifts, scales):
        for name, tensor in (("x", x), ("gate", gate), ("shift", shift), ("scale", scale)):
            if tensor.stride(-1) != 1 or tensor.shape[-1] != cols:
                raise ValueError(
                    f"{name} must be rows of {cols} contiguous columns, got {tensor.shape}, {tensor.stride()}."
                )
        segments.append((x, *(tensor.expand_as(x) for tensor in (gate, shift, scale))))
    for branch in branches:
        if branch.stride(-1) != 1:
            raise ValueError(f"Branch outputs must be contiguous in their last dimension, got {branch.stride()}.")
    if len(segments) == 1:
        segments.append(segments[0])
    branch0, branch1 = branches[0], branches[-1]
    if batch * rows_per_batch:
        _gated_residual_layernorm_modulate_kernel[(batch * rows_per_batch,)](
            *segments[0],
            *segments[1],
            branch0,
            branch1,
            hidden,
            out,
            fp4,
            quant_scale,
            xs[0].shape[1],
            rows_per_batch,
            cols,
            eps,
            *(stride for segment in segments for tensor in segment for stride in tensor.stride()[:2]),
            branch0.stride(0),
            branch0.stride(1),
            branch1.stride(0),
            branch1.stride(1),
            quant_scale.shape[1] if quantize else 0,
            TWO_BRANCHES=len(branches) == 2,
            QUANTIZE=quantize,
            BLOCK=triton.next_power_of_2(cols),
            num_warps=_ROW_NUM_WARPS,
        )
    return hidden, out, fp4, quant_scale


@torch.library.custom_op("xfuser::gated_residual_layernorm_modulate", mutates_args=())
def gated_residual_layernorm_modulate(
    xs: List[torch.Tensor],
    gates: List[torch.Tensor],
    shifts: List[torch.Tensor],
    scales: List[torch.Tensor],
    branches: List[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The joint hidden state ``x + gate * sum(branches)`` of the segments ``xs`` and its
    modulated norm ``(1 + scale) * layer_norm(.) + shift``, both ``[b, sum s_i, k]``.
    ``gates``, ``shifts`` and ``scales`` are broadcast-shaped views per segment."""
    hidden, out, _, _ = _gated_residual_layernorm_modulate(xs, gates, shifts, scales, branches, eps, quantize=False)
    return hidden, out


@gated_residual_layernorm_modulate.register_fake
def _(xs, gates, shifts, scales, branches, eps):
    return xs[0].new_empty(branches[0].shape), xs[0].new_empty(branches[0].shape)


@torch.library.custom_op("xfuser::gated_residual_layernorm_modulate_mxfp4_quant", mutates_args=())
def gated_residual_layernorm_modulate_mxfp4_quant(
    xs: List[torch.Tensor],
    gates: List[torch.Tensor],
    shifts: List[torch.Tensor],
    scales: List[torch.Tensor],
    branches: List[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``gated_residual_layernorm_modulate`` with the modulated norm returned as the MXFP4
    values and shuffled scales of its ``[b * sum s_i, k]`` rows (see ``mxfp4_fused_quant``)."""
    hidden, _, fp4, quant_scale = _gated_residual_layernorm_modulate(
        xs, gates, shifts, scales, branches, eps, quantize=True
    )
    return hidden, fp4, quant_scale


@gated_residual_layernorm_modulate_mxfp4_quant.register_fake
def _(xs, gates, shifts, scales, branches, eps):
    batch, rows, cols = branches[0].shape
    return (xs[0].new_empty(branches[0].shape), *mxfp4_fused_quant.mxfp4_buffers(batch * rows, cols, xs[0].device))
