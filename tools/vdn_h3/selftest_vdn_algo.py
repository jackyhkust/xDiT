#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU self-test for the eager VDN-H3 algorithm (no GPU, no big disk, no xfuser).

Loads only config.py, linear_branch.py and window_softmax.py by path -- with
stub parent packages so the real xfuser / diffusers imports never run -- then
exercises the window-softmax decomposition and the Video-Delta linear branch on
a tiny synthetic packed sequence, checking shapes, finiteness and the padding /
window invariants. This validates the ported math independent of the 62 GB model
materialization and the full pipeline.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKG = "xfuser.model_executor.models.transformers.vdn_minimax_h3"
PKG_DIR = os.path.join(
    REPO, "xfuser", "model_executor", "models", "transformers", "vdn_minimax_h3"
)


def _install_stub_packages() -> None:
    # Register empty parent packages so leaf modules' absolute imports resolve to
    # the stubs we load below, never triggering the heavy xfuser/diffusers init.
    parts = PKG.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


def _load_leaf(name: str) -> types.ModuleType:
    full = f"{PKG}.{name}"
    spec = importlib.util.spec_from_file_location(full, os.path.join(PKG_DIR, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import torch

    torch.manual_seed(0)
    _install_stub_packages()
    config = _load_leaf("config")
    linear_branch = _load_leaf("linear_branch")
    window_softmax = _load_leaf("window_softmax")

    hybrid = config.VDNHybridAttentionArchConfig(
        chunk=5, radius=1, anchor_frames="both", enable_softmax_gate=True,
        delta_rule="vdn_solve", linear_head_dim=8, bridge="alpha", a_fp32=True,
        enable_text_state=True, short_conv=("k", "v"),
    )

    heads, head_dim, hidden = 2, 8, 16
    num_frames, gh, gw = 20, 2, 2  # 20 frames -> window is partial, branch active
    tpf = gh * gw
    text_len, video_start = 3, 3
    video_end = video_start + num_frames * tpf
    used = video_end + 2  # two trailing audio rows
    seq_len = 128         # padded

    assert not hybrid.full_cover(num_frames), "expected a partial window for the test"

    layout = linear_branch.VDNH3Layout(
        seq_len=seq_len, used=used, text_len=text_len, video_start=video_start,
        num_frames=num_frames, tokens_per_frame=tpf, frame_height=gh, frame_width=gw,
    )
    plan = window_softmax.DecomposedPlan(layout, hybrid, torch.device("cpu"))
    print(f"[selftest] plan: dense_q={plan.dense_q.numel()} rows, "
          f"{len(plan.passes)} window pass(es); coverage OK")

    scale = head_dim ** -0.5
    q = torch.randn(seq_len, heads, head_dim)
    k = torch.randn(seq_len, heads, head_dim)
    v = torch.randn(seq_len, heads, head_dim)
    gate = torch.rand(seq_len, heads)

    window = window_softmax.windowed_softmax(
        q, k, v, plan=plan, layout=layout, scale=scale, softmax_gate=gate
    )
    assert window.shape == (seq_len, heads, head_dim), window.shape
    assert torch.isfinite(window[:used]).all(), "window has non-finite values"
    assert torch.count_nonzero(window[used:]) == 0, "padding rows must be zero"
    print(f"[selftest] windowed_softmax OK: {tuple(window.shape)}, "
          f"pad rows zeroed, finite")

    # Cross-check the dense-query rows against a plain full softmax.
    ref = torch.nn.functional.scaled_dot_product_attention(
        q[plan.dense_q].transpose(0, 1).unsqueeze(0),
        k[:used].transpose(0, 1).unsqueeze(0),
        v[:used].transpose(0, 1).unsqueeze(0),
        scale=scale,
    ).squeeze(0).transpose(0, 1)
    ref = ref * gate[plan.dense_q].unsqueeze(-1)
    err = (window[plan.dense_q] - ref).abs().max().item()
    assert err < 1e-4, f"dense-query rows disagree with reference softmax: {err}"
    print(f"[selftest] dense-query rows match reference softmax (max err {err:.2e})")

    branch = linear_branch.MiniMaxH3VDNLinearBranch(
        hybrid, hidden_size=hidden, num_attention_heads=heads, attention_head_dim=head_dim,
    ).float().eval()
    # short_conv / A_log / dt_bias are nn.Parameter(torch.empty(...)) meant to be
    # loaded from the checkpoint; give them finite values for the smoke test.
    with torch.no_grad():
        for p in branch.parameters():
            torch.nn.init.uniform_(p, -0.05, 0.05)

    x = torch.randn(seq_len, hidden)
    beta = branch.beta(x)
    out_gate = branch.output_gate(x)
    frame_mean = (
        x[video_start:video_end]
        .view(num_frames, tpf, hidden)
        .mean(dim=1, dtype=torch.float32)
    )
    vs = slice(video_start, video_end)
    ts = slice(0, text_len)
    q_raw = torch.randn(seq_len, heads, head_dim)
    k_raw = torch.randn(seq_len, heads, head_dim)
    v_raw = torch.randn(seq_len, heads, head_dim)

    with torch.no_grad():
        linear = branch(
            q_raw=q_raw[vs], k_raw=k_raw[vs], v_raw=v_raw[vs],
            beta=beta[vs], gate=out_gate[vs], frame_mean=frame_mean, layout=layout,
            text_k_raw=k_raw[ts], text_v_raw=v_raw[ts], text_beta=beta[ts],
        )
    exp_rows = num_frames * tpf
    assert linear.shape == (exp_rows, heads * head_dim), linear.shape
    assert torch.isfinite(linear).all(), "linear branch produced non-finite values"
    # anchor_frames="both": frames 0 and F-1 are handled by the dense softmax, so
    # the branch must leave those rows at zero.
    assert torch.count_nonzero(linear[:tpf]) == 0, "first frame must be zero (anchor)"
    assert torch.count_nonzero(linear[-tpf:]) == 0, "last frame must be zero (anchor)"
    assert torch.count_nonzero(linear[tpf:-tpf]) > 0, "interior frames should be active"
    print(f"[selftest] linear branch OK: {tuple(linear.shape)}, anchors zeroed, "
          f"interior active, finite")

    print("[selftest] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
