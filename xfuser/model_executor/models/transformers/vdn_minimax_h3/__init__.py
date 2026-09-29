# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 (Video DeltaNet MiniMax-H3) hybrid attention for xDiT, single-GPU eager.

This package ports OpenVDN/vdn-minimax-h3's hybrid attention (a chunk-aligned
frame-window softmax plus a bidirectional Video-Delta linear branch) onto xDiT's
MiniMax-H3 transformer. It reimplements the SGLang reference eagerly, with no
Ulysses/tensor-parallel sharding: ``local_heads == num_attention_heads`` and the
per-request metadata is built for the full local sequence.

The submodule/parameter names match the VDN overlay checkpoint's linear-branch
keys (``transformer_blocks.N.attn.{linear_attention.*, softmax_gate.*,
to_out_linear.weight}``) so the 800 branch tensors load with no remapping.
"""

from xfuser.model_executor.models.transformers.vdn_minimax_h3.config import (
    VDNHybridAttentionArchConfig,
)
from xfuser.model_executor.models.transformers.vdn_minimax_h3.attention import (
    VDNMiniMaxH3AttnProcessor,
    attach_vdn_modules,
    build_vdn_meta,
)
from xfuser.model_executor.models.transformers.vdn_minimax_h3.transformer import (
    xFuserMiniMaxH3VDNTransformer3DWrapper,
)

__all__ = [
    "VDNHybridAttentionArchConfig",
    "VDNMiniMaxH3AttnProcessor",
    "attach_vdn_modules",
    "build_vdn_meta",
    "xFuserMiniMaxH3VDNTransformer3DWrapper",
]
