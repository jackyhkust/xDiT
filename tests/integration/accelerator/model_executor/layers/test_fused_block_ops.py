"""The fused FLUX.3 block kernels must compute upstream's math.

They round once from fp32, as upstream does, but Triton's fp32 rsqrt, sums and fused
multiply-adds can differ from torch's in the last bit: in fp32 outputs that shows in most
values, in bf16 it rarely flips a rounding. The bf16 residual is exact; the quantized norm
must match the HIP quantization of the reference norm.
"""

import pytest
import torch
import torch.nn.functional as F

fused_block_ops = pytest.importorskip("xfuser.model_executor.layers.fused_block_ops")
from xfuser.model_executor.layers.usp import _seq_major_qkv_send  # noqa: E402

if not fused_block_ops.FUSED_BLOCK_OPS_AVAILABLE:
    pytest.skip("Triton is unavailable.", allow_module_level=True)

HEADS, HEAD_DIM, DIM, EPS = 24, 128, 3072, 1e-6


def _assert_rounds_like(actual, reference):
    """fp32: within a few ulps. bf16: at most one rounding step away, on few values."""
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
        return
    assert (actual != reference).float().mean().item() < 1e-3
    torch.testing.assert_close(actual, reference, rtol=2**-7, atol=1e-3)


def _qk_inputs(rows, dtype, generator):
    projected = (torch.randn(2, rows, 3 * DIM + 512, generator=generator, device="cuda") * 2).to(dtype)
    q, k, v = projected[..., :DIM], projected[..., DIM : 2 * DIM], projected[..., 2 * DIM : 3 * DIM]
    weights = [torch.rand(HEAD_DIM, generator=generator, device="cuda") + 0.5 for _ in range(2)]
    angles = torch.randn(rows, HEAD_DIM // 2, generator=generator, device="cuda", dtype=torch.float64) * 50
    cos = angles.cos().repeat_interleave(2, dim=1).float()
    sin = angles.sin().repeat_interleave(2, dim=1).float()
    return q, k, v, weights, cos, sin


def _upstream_qk(q, k, weights, cos, sin):
    """Flux3 attention's q/k norms and RoPE: fp32 math, one cast back."""
    from diffusers.models.embeddings import apply_rotary_emb

    out = []
    for x, weight in zip((q, k), weights):
        x = F.rms_norm(x.unflatten(-1, (HEADS, -1)).float(), (HEAD_DIM,), weight, EPS)
        out.append(apply_rotary_emb(x, (cos, sin), sequence_dim=1).to(q.dtype))
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("world_size", [1, 8])
@pytest.mark.parametrize("rows", [10, 522])
def test_qk_norm_rope_matches_upstream(rows, world_size, dtype):
    generator = torch.Generator(device="cuda").manual_seed(0)
    q, k, v, weights, cos, sin = _qk_inputs(rows, dtype, generator)
    ref_q, ref_k = _upstream_qk(q, k, weights, cos, sin)

    qk = torch.ops.xfuser.qk_norm_rope(q, k, *weights, EPS, cos, sin, HEADS)
    send = torch.ops.xfuser.qk_norm_rope_seq_major_send(q, k, v, *weights, EPS, cos, sin, HEADS, world_size)

    reference_send = _seq_major_qkv_send(ref_q, ref_k, v.unflatten(-1, (HEADS, -1)), world_size)
    torch.testing.assert_close(send[:, :, :, 2], reference_send[:, :, :, 2], rtol=0, atol=0)
    for fused, reference in ((qk[:, :, 0], ref_q), (qk[:, :, 1], ref_k), (send, reference_send)):
        _assert_rounds_like(fused, reference)


def _segments(lengths, dtype, generator):
    """Single-block style segments: each shares one modulation row (stride 0)."""
    segments = []
    for length in lengths:
        x = (torch.randn(2, length, DIM, generator=generator, device="cuda") * 4).to(dtype)
        mod = (torch.randn(2, 1, 3 * DIM, generator=generator, device="cuda") * 0.5).to(dtype)
        shift, scale, gate = mod.expand(-1, length, -1).chunk(3, dim=-1)
        segments.append((x, gate, shift, scale))
    return segments


def _reference(segments, attn, ff):
    hidden, normed, offset = [], [], 0
    for x, gate, shift, scale in segments:
        rows = slice(offset, offset + x.shape[1])
        h = (x.float() + gate.float() * (attn[:, rows].float() + ff[:, rows].float())).to(x.dtype)
        hidden.append(h)
        normed.append((1 + scale.float()) * F.layer_norm(h.float(), (DIM,), eps=EPS) + shift.float())
        offset += x.shape[1]
    return torch.cat(hidden, dim=1), torch.cat(normed, dim=1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lengths", [(522,), (10, 512), (0, 66)])
def test_gated_residual_layernorm_modulate_matches_upstream(lengths, dtype):
    generator = torch.Generator(device="cuda").manual_seed(0)
    segments = _segments(lengths, dtype, generator)
    attn, ff = (torch.randn(2, sum(lengths), DIM, generator=generator, device="cuda").to(dtype) for _ in range(2))
    ref_hidden, ref_normed = _reference(segments, attn, ff)

    xs, gates, shifts, scales = (list(column) for column in zip(*segments))
    hidden, normed = torch.ops.xfuser.gated_residual_layernorm_modulate(xs, gates, shifts, scales, attn, ff, EPS)

    if dtype == torch.bfloat16:
        torch.testing.assert_close(hidden, ref_hidden, rtol=0, atol=0)
    else:
        _assert_rounds_like(hidden, ref_hidden)
    _assert_rounds_like(normed, ref_normed.to(dtype))


@pytest.mark.parametrize("lengths", [(522,), (10, 2048)])
def test_gated_residual_quant_matches_hip_quant_of_the_reference(lengths):
    aiter = pytest.importorskip("aiter")
    from xfuser.model_executor.layers import mxfp4_fused_quant

    if not mxfp4_fused_quant.MXFP4_FUSED_QUANT_AVAILABLE:
        pytest.skip("AITER's Triton MXFP4 helpers are unavailable.")
    generator = torch.Generator(device="cuda").manual_seed(0)
    segments = _segments(lengths, torch.bfloat16, generator)
    attn, ff = (torch.randn(2, sum(lengths), DIM, generator=generator, device="cuda").bfloat16() for _ in range(2))
    ref_hidden, ref_normed = _reference(segments, attn, ff)
    ref_fp4, ref_scale = aiter.get_hip_quant(aiter.QuantType.per_1x32)(
        ref_normed.to(torch.bfloat16).flatten(0, 1).contiguous(), shuffle=True
    )

    xs, gates, shifts, scales = (list(column) for column in zip(*segments))
    hidden, fp4, scale = torch.ops.xfuser.gated_residual_layernorm_modulate_mxfp4_quant(
        xs, gates, shifts, scales, attn, ff, EPS
    )

    torch.testing.assert_close(hidden, ref_hidden, rtol=0, atol=0)
    rows = 2 * sum(lengths)
    x = torch.arange(rows, device="cuda")[:, None]
    y = torch.arange(DIM // 32, device="cuda")[None, :]
    columns = scale.shape[1]
    positions = (
        (x // 32 * columns) * 32 + (y // 8) * 256 + (y % 4) * 64 + (x % 16) * 4 + (y % 8) // 4 * 2 + (x % 32) // 16
    )
    assert (fp4 != ref_fp4.view(torch.uint8)).float().mean().item() < 1e-3
    written = positions.flatten()
    assert (scale.flatten()[written] != ref_scale.view(torch.uint8).flatten()[written]).float().mean().item() < 1e-3
