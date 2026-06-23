import copy
import json
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.model_executor.models.transformers.transformer_ideogram4 import (
    get_ideogram4_transformer_wrapper_class,
)
from xfuser.model_executor.pipelines.pipeline_ideogram4 import (
    get_ideogram4_pipeline_class,
)
from xfuser.model_executor.models.runner_models.base_model import (
    ModelSettings,
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
)
from xfuser.core.distributed.parallel_state import get_vae_parallel_group
from xfuser.core.utils.runner_utils import log


# --- Minimal FP8 weight loading (from ideogram-oss/ideogram4) ---

FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"


class Fp8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias, compute_dtype):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def _is_fp8_state_dict(state_dict):
    return any(k.endswith(FP8_SCALE_SUFFIX) for k in state_dict) or any(
        v.dtype == FP8_WEIGHT_DTYPE for v in state_dict.values()
    )


def _swap_linears_to_fp8(module, state_dict, compute_dtype, prefix=""):
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}{name}"
        if isinstance(child, nn.Linear) and f"{child_prefix}{FP8_SCALE_SUFFIX}" in state_dict:
            setattr(module, name, Fp8Linear(
                child.in_features, child.out_features,
                bias=child.bias is not None, compute_dtype=compute_dtype,
            ))
        else:
            _swap_linears_to_fp8(child, state_dict, compute_dtype, prefix=f"{child_prefix}.")


def _load_fp8_state_dict(model, state_dict, device, dtype):
    prepared = {}
    for k, v in state_dict.items():
        if v.dtype == FP8_WEIGHT_DTYPE:
            prepared[k] = v.to(device=device)
        elif k.endswith(FP8_SCALE_SUFFIX):
            prepared[k] = v.to(device=device, dtype=torch.float32)
        elif v.is_floating_point():
            prepared[k] = v.to(device=device, dtype=dtype)
        else:
            prepared[k] = v.to(device=device)
    missing, unexpected = model.load_state_dict(prepared, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in FP8 state dict: {unexpected[:10]}")
    if missing:
        warnings.warn(f"Missing keys in FP8 state dict: {missing[:10]}")
    model.to(device)


def _load_sharded_safetensors(repo_id, subfolder, basename="diffusion_pytorch_model"):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    from safetensors.torch import load_file
    index_file = f"{subfolder}/{basename}.safetensors.index.json"
    try:
        index_path = hf_hub_download(repo_id=repo_id, filename=index_file)
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shard_filenames = sorted(set(weight_map.values()))
        state_dict = {}
        for shard in shard_filenames:
            shard_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/{shard}")
            state_dict.update(load_file(shard_path))
        return state_dict
    except EntryNotFoundError:
        single_file = f"{subfolder}/{basename}.safetensors"
        path = hf_hub_download(repo_id=repo_id, filename=single_file)
        return load_file(path)


def _convert_ideogram_to_diffusers_keys(state_dict):
    """Convert ideogram-oss FP8 key names to diffusers key names.

    Main differences:
    - attention.qkv.weight [3H, D] -> attention.to_q/to_k/to_v.weight [H, D]
    - attention.o.weight -> attention.to_out.0.weight
    - Same for weight_scale tensors
    """
    converted = {}
    for key, tensor in state_dict.items():
        if key.endswith(".attention.qkv.weight_scale"):
            base = key.removesuffix(".attention.qkv.weight_scale")
            q, k, v = tensor.chunk(3, dim=0)
            converted[f"{base}.attention.to_q{FP8_SCALE_SUFFIX}"] = q
            converted[f"{base}.attention.to_k{FP8_SCALE_SUFFIX}"] = k
            converted[f"{base}.attention.to_v{FP8_SCALE_SUFFIX}"] = v
        elif key.endswith(".attention.qkv.weight"):
            base = key.removesuffix(".attention.qkv.weight")
            q, k, v = tensor.chunk(3, dim=0)
            converted[f"{base}.attention.to_q.weight"] = q
            converted[f"{base}.attention.to_k.weight"] = k
            converted[f"{base}.attention.to_v.weight"] = v
        elif key.endswith(".attention.o.weight_scale"):
            converted[key.replace(".attention.o.weight_scale", ".attention.to_out.0.weight_scale")] = tensor
        elif key.endswith(".attention.o.weight"):
            converted[key.replace(".attention.o.weight", ".attention.to_out.0.weight")] = tensor
        else:
            converted[key] = tensor
    return converted


# --- End FP8 loading ---


def _setup_parallel_vae(vae):
    try:
        from distvae.modules.adapters.vae.decoder_adapters import DecoderAdapter
        patched_decoder = DecoderAdapter(
            vae.decoder, vae_group=get_vae_parallel_group().device_group
        ).to(vae.device)
        vae.decoder = patched_decoder
        log("Parallel VAE decoder enabled.")
    except ImportError:
        log("DistVAE not available for decoder. Defaulting to single-rank.")
    except Exception as e:
        raise ValueError(f"Failed to patch VAE decoder: {e}")


def _detect_fp8_weights(model_id, subfolder="transformer"):
    from huggingface_hub import hf_hub_download
    try:
        cfg_path = hf_hub_download(repo_id=model_id, filename=f"{subfolder}/config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        # No quantization_config means it could be FP8 weight-only (ideogram format)
        # or plain BF16. Check by looking at the weight files.
        if "quantization_config" in cfg:
            return cfg["quantization_config"].get("quant_method") == "fp8"
    except Exception:
        pass
    # Heuristic: try loading a small part of the state dict to check dtype
    try:
        state_dict = _load_sharded_safetensors(model_id, subfolder)
        return _is_fp8_state_dict(state_dict)
    except Exception:
        return False


@register_model("ideogram-ai/ideogram-4-nf4")
@register_model("ideogram-ai/ideogram-4-nf4-diffusers")
@register_model("ideogram-ai/ideogram-4-fp8")
@register_model("CalamitousFelicitousness/Ideogram-4-bf16-Diffusers")
@register_model("Ideogram-4")
class xFuserIdeogram4Model(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_cfg_parallel=True,
        use_fp8_gemms=True,
        use_fp4_gemms=True,
        fully_shard_degree=True,
        use_parallel_vae=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    default_input_values = DefaultInputValues(
        height=2048,
        width=2048,
        num_inference_steps=48,
        guidance_scale=7.0,
    )
    settings = ModelSettings(
        model_name="ideogram-ai/ideogram-4-nf4",
        output_name="ideogram4",
        model_output_type="image",
        fp8_gemm_module_list=[
            "transformer.layers",
            "unconditional_transformer.layers",
        ],
        fp4_gemm_module_list=[
            "transformer.layers",
            "unconditional_transformer.layers",
        ],
        fsdp_strategy={
            "transformer": {"wrap_attrs": ["layers"]},
            "unconditional_transformer": {"wrap_attrs": ["layers"]},
        },
    )

    def _load_fp8_transformer(self, model_id, subfolder, device):
        xFuserTransformer = get_ideogram4_transformer_wrapper_class()
        model = xFuserTransformer.from_config(
            xFuserTransformer.load_config(model_id, subfolder=subfolder)
        )
        model.__class__ = xFuserTransformer
        model._install_xfuser_processors()
        state_dict = _load_sharded_safetensors(model_id, subfolder)
        state_dict = _convert_ideogram_to_diffusers_keys(state_dict)
        model.to(torch.bfloat16)
        _swap_linears_to_fp8(model, state_dict, compute_dtype=torch.bfloat16)
        _load_fp8_state_dict(model, state_dict, device=device, dtype=torch.bfloat16)
        model.eval()
        log(f"Loaded FP8 transformer from {model_id}/{subfolder}")
        return model

    def _load_fp8_text_encoder(self, model_id, device):
        """Load FP8 text encoder using custom weight-only FP8 path."""
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(
            model_id, subfolder="text_encoder", trust_remote_code=True
        )
        model = AutoModel.from_config(config, trust_remote_code=True)
        state_dict = _load_sharded_safetensors(model_id, "text_encoder", basename="model")
        _swap_linears_to_fp8(model, state_dict, compute_dtype=torch.bfloat16)
        _load_fp8_state_dict(model, state_dict, device=device, dtype=torch.bfloat16)
        model.eval()
        log(f"Loaded FP8 text encoder from {model_id}/text_encoder")
        return model

    def _load_model(self) -> DiffusionPipeline:
        model_id = self.config.model
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")

        is_fp8 = _detect_fp8_weights(model_id)

        if is_fp8:
            log(f"Detected FP8 weight format, using custom FP8 loader")
            transformer = self._load_fp8_transformer(model_id, "transformer", device)
            unconditional_transformer = self._load_fp8_transformer(
                model_id, "unconditional_transformer", device
            )
            text_encoder = self._load_fp8_text_encoder(model_id, device)
        else:
            xFuserTransformer = get_ideogram4_transformer_wrapper_class()
            transformer = xFuserTransformer.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, subfolder="transformer",
            )
            unconditional_transformer = xFuserTransformer.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, subfolder="unconditional_transformer",
            )
            text_encoder = None

        load_kwargs = dict(
            pretrained_model_name_or_path=model_id,
            transformer=transformer,
            unconditional_transformer=unconditional_transformer,
            torch_dtype=torch.bfloat16,
        )
        if text_encoder is not None:
            load_kwargs["text_encoder"] = text_encoder

        # Load prompt enhancer head for automatic JSON caption conversion
        try:
            from diffusers.pipelines.ideogram4.prompt_enhancer import (
                Ideogram4PromptEnhancerHead,
            )
            prompt_enhancer_head = Ideogram4PromptEnhancerHead.from_pretrained(
                "diffusers/qwen3-vl-8b-instruct-lm-head",
                torch_dtype=torch.bfloat16,
            )
            load_kwargs["prompt_enhancer_head"] = prompt_enhancer_head
            log("Loaded prompt enhancer head for automatic JSON caption conversion")
        except Exception as e:
            log(f"Prompt enhancer head not available ({e}), plain text prompts may produce poor results")

        xFuserPipeline = get_ideogram4_pipeline_class()
        pipe = xFuserPipeline.from_pretrained(**load_kwargs)
        return pipe

    def _is_json_prompt(self, prompt):
        if not isinstance(prompt, str):
            return False
        stripped = prompt.strip()
        return stripped.startswith("{") and stripped.endswith("}")

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        prompt = input_args["prompt"]
        use_upsampling = not self._is_json_prompt(prompt) and hasattr(self.pipe, "prompt_enhancer_head") and self.pipe.prompt_enhancer_head is not None

        output = self.pipe(
            prompt=prompt,
            height=input_args["height"],
            width=input_args["width"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            guidance_schedule=None,
            prompt_upsampling=use_upsampling,
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
            output_type="pil",
        )
        return DiffusionOutput(images=output.images, pipe_args=input_args)

    def _compile_model(self, input_args: dict) -> None:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")
        self.pipe.unconditional_transformer = torch.compile(
            self.pipe.unconditional_transformer, mode="default"
        )
        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = 2
        self._run_timed_pipe(compile_args)

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        # FP8 transformers are already on device; skip the pipe.to() for them
        has_fp8 = any(isinstance(m, Fp8Linear)
                      for m in self.pipe.transformer.modules())
        if has_fp8:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = f"cuda:{local_rank}"
            for name, component in self.pipe.components.items():
                if name in ("transformer", "unconditional_transformer"):
                    continue
                if component is not None and hasattr(component, "to"):
                    component.to(device)
        else:
            super()._post_load_and_state_initialization(input_args)
        if self.config.use_parallel_vae:
            _setup_parallel_vae(self.pipe.vae)
