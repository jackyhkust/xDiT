"""MXFP4 activation quantization fused into the kernels that produce FP4 GEMM inputs.

Each op returns the packed E2M1 values and shuffled E8M0 scales that AITER's HIP per-1x32
quantization (``aiter.get_hip_quant(per_1x32)(y, shuffle=True)``, the one inside
``xfuser::mxfp4_gemm``) gives for the producer's bf16 output ``y``, so
``xfuser::mxfp4_gemm_quantized`` takes them as they are and ``y`` is never written:

- ``layernorm_modulate_mxfp4_quant``: ``(1 + scale) * layer_norm(x) + shift``, a FLUX.3 block input.
- ``swiglu_mxfp4_quant``: ``silu(x[..., :n]) * x[..., n:]``, the feed-forward hidden state.
- ``seq_major_attention_output_mxfp4_quant``: the ``[b * s, h * d]`` rows of a Ulysses output
  all-to-all receive buffer ``[P, s, b, h / P, d]``, without the reorder copy.
- ``attention_output_swiglu_mxfp4_quant``: those attention rows and the feed-forward hidden
  state side by side, the input of a block's merged output GEMM.

Both outputs are uint8: the values are ``[M, K / 2]`` and the scales are
``[ceil(M / 256) * 256, ceil(K / 256) * 8]`` in the order of ``_mx_scale_index``. The
producers keep fp32 intermediates and round once to bf16 before quantizing, as Inductor's
fused kernels do.
"""

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_pack_op

    MXFP4_FUSED_QUANT_AVAILABLE = True
except ImportError:
    MXFP4_FUSED_QUANT_AVAILABLE = False

# Columns per program (a multiple of the 32-column group) and warps: the fastest pair for
# all three kernels at FLUX.3's shapes on MI355X (M 522 and 2058).
_BLOCK_K = 1024
_NUM_WARPS = 2


def mxfp4_buffers(rows: int, cols: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uninitialized value and scale buffers for an ``[rows, cols]`` activation, shaped as
    the HIP quantization allocates them."""
    if cols % 32:
        raise ValueError(f"MXFP4 quantizes groups of 32 columns, got {cols} columns.")
    fp4 = torch.empty(rows, cols // 2, dtype=torch.uint8, device=device)
    scale = torch.empty((rows + 255) // 256 * 256, (cols // 32 + 7) // 8 * 8, dtype=torch.uint8, device=device)
    return fp4, scale


if MXFP4_FUSED_QUANT_AVAILABLE:

    @triton.jit
    def _mx_scale_index(row, group, scale_cols):
        """The HIP quantization's position for the scale of ``group`` in ``row``."""
        return (
            (row // 32 * scale_cols) * 32
            + (group // 8) * 256
            + (group % 4) * 64
            + (row % 16) * 4
            + (group % 8) // 4 * 2
            + (row % 32) // 16
        )

    @triton.jit
    def _store_mxfp4_row(y, row, first_col, cols, fp4_ptr, scale_ptr, scale_cols, BLOCK_K: tl.constexpr):
        """Quantize ``y``, fp32 ``[BLOCK_K]`` holding bf16 values for columns ``first_col``
        onwards of ``row``, by the HIP rule: the dequantization scale is
        ``ceil_pow2(max(amax, 1e-10) * fp32(1 / 6))`` and the values round to E2M1 after a
        multiply by its exact reciprocal. Columns at or past ``cols`` must hold zero."""
        _store_mxfp4_row_part(y, row, first_col, cols, 0, cols, fp4_ptr, scale_ptr, scale_cols, BLOCK_K)

    @triton.jit
    def _store_mxfp4_row_part(
        y, row, first_col, cols, col_offset, row_cols, fp4_ptr, scale_ptr, scale_cols, BLOCK_K: tl.constexpr
    ):
        """``_store_mxfp4_row`` into columns ``col_offset`` to ``col_offset + cols`` of rows of
        ``row_cols`` columns; ``col_offset`` is a multiple of the 32-column group."""
        GROUPS: tl.constexpr = BLOCK_K // 32
        groups = tl.reshape(y, [1, GROUPS, 32])
        amax = tl.maximum(tl.max(tl.abs(groups), axis=2), 1e-10)
        bits = (amax * 0.16666666666666666).to(tl.uint32, bitcast=True)
        exponent = (bits >> 23) & 0xFF
        exponent = tl.where((exponent < 0xFF) & ((bits & 0x7FFFFF) != 0), exponent + 1, exponent)
        inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
        packed = _mxfp4_pack_op(tl.reshape(groups * inverse[:, :, None], [1, BLOCK_K]), BLOCK_K, 1, 32)

        half_cols = first_col // 2 + tl.arange(0, BLOCK_K // 2)
        tl.store(
            fp4_ptr + row.to(tl.int64) * (row_cols // 2) + col_offset // 2 + half_cols[None, :],
            packed,
            mask=half_cols[None, :] < cols // 2,
        )
        group = first_col // 32 + tl.arange(0, GROUPS)
        tl.store(
            scale_ptr + _mx_scale_index(row, col_offset // 32 + group[None, :], scale_cols),
            exponent.to(tl.uint8),
            mask=group[None, :] < cols // 32,
        )

    @triton.jit
    def _layernorm_modulate_mxfp4_kernel(
        x_ptr,
        shift_ptr,
        mod_scale_ptr,
        fp4_ptr,
        scale_ptr,
        seq_len,
        out_rows_per_batch,
        out_row_offset,
        cols,
        eps,
        x_stride_b,
        x_stride_s,
        shift_stride_b,
        shift_stride_s,
        mod_scale_stride_b,
        mod_scale_stride_s,
        scale_cols,
        BLOCK_K: tl.constexpr,
    ):
        index = tl.program_id(0)
        b = index // seq_len
        s = index % seq_len
        x_row = x_ptr + b.to(tl.int64) * x_stride_b + s.to(tl.int64) * x_stride_s
        shift_row = shift_ptr + b.to(tl.int64) * shift_stride_b + s.to(tl.int64) * shift_stride_s
        mod_scale_row = mod_scale_ptr + b.to(tl.int64) * mod_scale_stride_b + s.to(tl.int64) * mod_scale_stride_s
        offsets = tl.arange(0, BLOCK_K)

        total = tl.zeros([BLOCK_K], dtype=tl.float32)
        for start in range(0, cols, BLOCK_K):
            total += tl.load(x_row + start + offsets, mask=start + offsets < cols, other=0.0).to(tl.float32)
        mean = tl.sum(total) / cols
        squares = tl.zeros([BLOCK_K], dtype=tl.float32)
        for start in range(0, cols, BLOCK_K):
            mask = start + offsets < cols
            centered = tl.load(x_row + start + offsets, mask=mask, other=0.0).to(tl.float32) - mean
            centered = tl.where(mask, centered, 0.0)
            squares += centered * centered
        rstd = tl.rsqrt(tl.sum(squares) / cols + eps)

        row = b * out_rows_per_batch + out_row_offset + s
        for start in range(0, cols, BLOCK_K):
            mask = start + offsets < cols
            x = tl.load(x_row + start + offsets, mask=mask, other=0.0).to(tl.float32)
            shift = tl.load(shift_row + start + offsets, mask=mask, other=0.0).to(tl.float32)
            mod_scale = tl.load(mod_scale_row + start + offsets, mask=mask, other=0.0).to(tl.float32)
            y = (1.0 + mod_scale) * ((x - mean) * rstd) + shift
            y = tl.where(mask, y.to(tl.bfloat16).to(tl.float32), 0.0)
            _store_mxfp4_row(y, row, start, cols, fp4_ptr, scale_ptr, scale_cols, BLOCK_K)

    @triton.jit
    def _swiglu_mxfp4_kernel(
        x_ptr, fp4_ptr, scale_ptr, cols, col_offset, row_cols, x_stride_row, scale_cols, BLOCK_K: tl.constexpr
    ):
        row = tl.program_id(0)
        start = tl.program_id(1) * BLOCK_K
        col = start + tl.arange(0, BLOCK_K)
        mask = col < cols
        x_row = x_ptr + row.to(tl.int64) * x_stride_row
        gate = tl.load(x_row + col, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_row + cols + col, mask=mask, other=0.0).to(tl.float32)
        y = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16).to(tl.float32)
        _store_mxfp4_row_part(y, row, start, cols, col_offset, row_cols, fp4_ptr, scale_ptr, scale_cols, BLOCK_K)

    @triton.jit
    def _seq_major_output_mxfp4_kernel(
        x_ptr,
        fp4_ptr,
        scale_ptr,
        seq_len,
        cols,
        col_offset,
        row_cols,
        chunk_cols,
        stride_rank,
        stride_s,
        stride_b,
        scale_cols,
        descale_ptr,
        DEQUANT: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        start = tl.program_id(1) * BLOCK_K
        col = start + tl.arange(0, BLOCK_K)
        mask = col < cols
        b = row // seq_len
        s = row % seq_len
        offsets = (
            b.to(tl.int64) * stride_b
            + s.to(tl.int64) * stride_s
            + (col // chunk_cols).to(tl.int64) * stride_rank
            + col % chunk_cols
        )
        y = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if DEQUANT:
            # The FP8 exchange's dequantization, (x.float() * descale).to(bf16).
            y = (y * tl.load(descale_ptr)).to(tl.bfloat16).to(tl.float32)
        _store_mxfp4_row_part(y, row, start, cols, col_offset, row_cols, fp4_ptr, scale_ptr, scale_cols, BLOCK_K)


def _check_rows(name: str, tensor: torch.Tensor) -> None:
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous in its last dimension, got strides {tensor.stride()}.")


@torch.library.custom_op("xfuser::layernorm_modulate_mxfp4_quant", mutates_args=())
def layernorm_modulate_mxfp4_quant(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    x_after: Optional[torch.Tensor] = None,
    shift_after: Optional[torch.Tensor] = None,
    scale_after: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The quantized ``(1 + scale) * layer_norm(x) + shift`` of ``[b, s, k]`` ``x``, with
    ``shift`` and ``scale`` broadcast-shaped views (stride 0 where shared). With ``x_after``,
    each batch row's sequence continues with ``x_after``'s, modulated by its own pair, as if
    the two results were concatenated along the sequence."""
    segments = [(x, shift, scale)]
    if x_after is not None:
        segments.append((x_after, shift_after, scale_after))
    batch, cols = x.shape[0], x.shape[-1]
    rows_per_batch = sum(segment[0].shape[1] for segment in segments)
    fp4, quant_scale = mxfp4_buffers(batch * rows_per_batch, cols, x.device)
    offset = 0
    for x_part, shift_part, scale_part in segments:
        seq_len = x_part.shape[1]
        for name, tensor in (("x", x_part), ("shift", shift_part), ("scale", scale_part)):
            _check_rows(name, tensor)
            if tensor.shape[-1] != cols:
                raise ValueError(f"{name} has {tensor.shape[-1]} columns, expected {cols}.")
        shift_part = shift_part.expand_as(x_part)
        scale_part = scale_part.expand_as(x_part)
        if batch * seq_len:
            _layernorm_modulate_mxfp4_kernel[(batch * seq_len,)](
                x_part,
                shift_part,
                scale_part,
                fp4,
                quant_scale,
                seq_len,
                rows_per_batch,
                offset,
                cols,
                eps,
                x_part.stride(0),
                x_part.stride(1),
                shift_part.stride(0),
                shift_part.stride(1),
                scale_part.stride(0),
                scale_part.stride(1),
                quant_scale.shape[1],
                BLOCK_K=_BLOCK_K,
                num_warps=_NUM_WARPS,
            )
        offset += seq_len
    return fp4, quant_scale


@layernorm_modulate_mxfp4_quant.register_fake
def _(x, shift, scale, eps, x_after=None, shift_after=None, scale_after=None):
    rows = x.shape[1] + (0 if x_after is None else x_after.shape[1])
    return mxfp4_buffers(x.shape[0] * rows, x.shape[-1], x.device)


def _swiglu_rows(x: torch.Tensor) -> torch.Tensor:
    _check_rows("x", x)
    return x.flatten(0, -2) if x.ndim > 2 else x


def _launch_swiglu(rows, fp4, quant_scale, col_offset):
    """Quantize the SwiGLU of ``[M, 2n]`` ``rows`` into columns ``col_offset`` onwards."""
    cols = rows.shape[1] // 2
    if rows.shape[0]:
        _swiglu_mxfp4_kernel[(rows.shape[0], triton.cdiv(cols, _BLOCK_K))](
            rows,
            fp4,
            quant_scale,
            cols,
            col_offset,
            fp4.shape[1] * 2,
            rows.stride(0),
            quant_scale.shape[1],
            BLOCK_K=_BLOCK_K,
            num_warps=_NUM_WARPS,
        )


def _received_cols(received: torch.Tensor) -> int:
    world_size, _, _, heads, head_dim = received.shape
    if received.stride(4) != 1 or received.stride(3) != head_dim:
        raise ValueError(f"Each received chunk must hold contiguous heads, got strides {received.stride()}.")
    return world_size * heads * head_dim


def _launch_seq_major_output(received, fp4, quant_scale, col_offset, descale=None):
    """Quantize the rows of a ``[P, s, b, h, d]`` receive buffer into columns ``col_offset`` onwards,
    an FP8 one dequantized by its one-element ``descale`` first."""
    _, seq_len, batch, heads, head_dim = received.shape
    cols = _received_cols(received)
    if (descale is None) != (received.element_size() != 1):
        raise ValueError(f"A descale goes with an FP8 receive buffer, got {received.dtype} and descale {descale}.")
    if batch * seq_len:
        _seq_major_output_mxfp4_kernel[(batch * seq_len, triton.cdiv(cols, _BLOCK_K))](
            received,
            fp4,
            quant_scale,
            seq_len,
            cols,
            col_offset,
            fp4.shape[1] * 2,
            heads * head_dim,
            received.stride(0),
            received.stride(1),
            received.stride(2),
            quant_scale.shape[1],
            received if descale is None else descale,
            DEQUANT=descale is not None,
            BLOCK_K=_BLOCK_K,
            num_warps=_NUM_WARPS,
        )


@torch.library.custom_op("xfuser::swiglu_mxfp4_quant", mutates_args=())
def swiglu_mxfp4_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The quantized ``silu(x[..., :n]) * x[..., n:]`` rows of ``x``, ``[..., 2n]``; the
    rows may be strided, e.g. a column slice of a wider GEMM output."""
    rows = _swiglu_rows(x)
    fp4, quant_scale = mxfp4_buffers(rows.shape[0], rows.shape[1] // 2, x.device)
    _launch_swiglu(rows, fp4, quant_scale, 0)
    return fp4, quant_scale


@swiglu_mxfp4_quant.register_fake
def _(x):
    return mxfp4_buffers(x.numel() // x.shape[-1], x.shape[-1] // 2, x.device)


@torch.library.custom_op("xfuser::seq_major_attention_output_mxfp4_quant", mutates_args=())
def seq_major_attention_output_mxfp4_quant(
    received: torch.Tensor, descale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The quantized ``[b * s, P * h * d]`` rows of a ``[P, s, b, h, d]`` output receive
    buffer, chunk ``p`` holding heads ``p * h`` onwards for this rank's ``s`` tokens. An FP8
    buffer (FP8 Ulysses comms) comes with its one-element ``descale``."""
    _, seq_len, batch, _, _ = received.shape
    fp4, quant_scale = mxfp4_buffers(batch * seq_len, _received_cols(received), received.device)
    _launch_seq_major_output(received, fp4, quant_scale, 0, descale)
    return fp4, quant_scale


@seq_major_attention_output_mxfp4_quant.register_fake
def _(received, descale=None):
    world_size, seq_len, batch, heads, head_dim = received.shape
    return mxfp4_buffers(batch * seq_len, world_size * heads * head_dim, received.device)


@torch.library.custom_op("xfuser::attention_output_swiglu_mxfp4_quant", mutates_args=())
def attention_output_swiglu_mxfp4_quant(
    received: torch.Tensor, ff_hidden: torch.Tensor, descale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The quantized ``[b * s, P * h * d + n]`` rows of ``torch.cat([attention, swiglu],
    dim=-1)``: the ``received`` rows as ``seq_major_attention_output_mxfp4_quant`` reads
    them (with ``descale`` for an FP8 buffer), then ``swiglu_mxfp4_quant`` of the matching
    ``[b, s, 2n]`` ``ff_hidden`` rows. The attention width must be a multiple of the
    32-column group."""
    _, seq_len, batch, _, _ = received.shape
    attention_cols = _received_cols(received)
    if attention_cols % 32:
        raise ValueError(f"The attention rows must fill whole 32-column groups, got {attention_cols} columns.")
    rows = _swiglu_rows(ff_hidden)
    if rows.shape[0] != batch * seq_len:
        raise ValueError(f"ff_hidden has {rows.shape[0]} rows, the attention output {batch * seq_len}.")
    fp4, quant_scale = mxfp4_buffers(batch * seq_len, attention_cols + rows.shape[1] // 2, received.device)
    _launch_seq_major_output(received, fp4, quant_scale, 0, descale)
    _launch_swiglu(rows, fp4, quant_scale, attention_cols)
    return fp4, quant_scale


@attention_output_swiglu_mxfp4_quant.register_fake
def _(received, ff_hidden, descale=None):
    world_size, seq_len, batch, heads, head_dim = received.shape
    cols = world_size * heads * head_dim + ff_hidden.shape[-1] // 2
    return mxfp4_buffers(batch * seq_len, cols, received.device)
