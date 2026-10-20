"""Load a hub model from the local HF cache without any network call.

With ``HF_HUB_OFFLINE=1``, loading MiniMax-H3 by repo id still fails even when
the cache is complete: diffusers' sharded-checkpoint loader calls the hub
``model_info`` API for a repo id, and transformers probes for files the repo
never had (e.g. ``processor/config.json``) and treats the cache miss as fatal.
Both paths are skipped when components load from a local directory, so this
resolves the repo id to its cached snapshot and points every modular-pipeline
component spec at it.
"""

from __future__ import annotations


def resolve_local_snapshot(repo_id: str) -> str | None:
    """The cached snapshot directory for ``repo_id``, or None if it is not cached."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return None


def localize_modular_components(repo_id: str, local_dir: str) -> None:
    """Make ``ModularPipeline.load_components`` read ``repo_id`` components from ``local_dir``."""
    from diffusers import ModularPipeline

    if getattr(ModularPipeline, "_xfuser_localized", None) == (repo_id, local_dir):
        return
    original_load_components = getattr(ModularPipeline, "_xfuser_original_load_components", None)
    if original_load_components is None:
        original_load_components = ModularPipeline.load_components
        ModularPipeline._xfuser_original_load_components = original_load_components

    def load_components(self, *args, **kwargs):
        for spec in self._component_specs.values():
            if spec.pretrained_model_name_or_path == repo_id:
                spec.pretrained_model_name_or_path = local_dir
        return original_load_components(self, *args, **kwargs)

    ModularPipeline.load_components = load_components
    ModularPipeline._xfuser_localized = (repo_id, local_dir)
