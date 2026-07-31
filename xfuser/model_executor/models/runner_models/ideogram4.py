import copy
import json
import os

import torch
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
from xfuser.core.utils.runner_utils import log


# --- Minimal FP8 weight loading (from ideogram-oss/ideogram4) ---

FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"


def _is_fp8_state_dict(state_dict):
    return any(k.endswith(FP8_SCALE_SUFFIX) for k in state_dict) or any(
        v.dtype == FP8_WEIGHT_DTYPE for v in state_dict.values()
    )


def _dequantize_fp8_state_dict(state_dict, dtype=torch.bfloat16):
    """Dequantize FP8 weight+scale pairs into standard dtype tensors.

    Converts `key.weight` (float8) + `key.weight_scale` (float32) pairs
    into a single `key.weight` tensor in the target dtype, and drops
    the scale keys. Non-FP8 tensors are cast to dtype as-is.
    """
    result = {}
    scale_keys = {k for k in state_dict if k.endswith(FP8_SCALE_SUFFIX)}
    for key, tensor in state_dict.items():
        if key in scale_keys:
            continue
        scale_key = key + "_scale"
        if tensor.dtype == FP8_WEIGHT_DTYPE and scale_key in state_dict:
            scale = state_dict[scale_key].to(torch.float32)
            result[key] = (tensor.to(torch.float32) * scale.unsqueeze(-1)).to(dtype)
        elif tensor.is_floating_point():
            result[key] = tensor.to(dtype)
        else:
            result[key] = tensor
    return result


def _load_sharded_safetensors(repo_id, subfolder, basename="diffusion_pytorch_model"):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    from safetensors.torch import load_file
    index_file = f"{subfolder}/{basename}.safetensors.index.json"
    # Only the index download signals sharded-vs-single. Scope the fallback to
    # that probe so a genuinely missing shard (below) surfaces instead of being
    # masked as "not sharded".
    try:
        index_path = hf_hub_download(repo_id=repo_id, filename=index_file)
    except EntryNotFoundError:
        single_file = f"{subfolder}/{basename}.safetensors"
        path = hf_hub_download(repo_id=repo_id, filename=single_file)
        return load_file(path)
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_filenames = sorted(set(weight_map.values()))
    state_dict = {}
    for shard in shard_filenames:
        shard_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/{shard}")
        state_dict.update(load_file(shard_path))
    return state_dict


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
        use_hybrid_gemm_schedule=True,
        fully_shard_degree=True,
        # Parallel VAE intentionally omitted: A/B tested on 4-GPU 2048^2 (no
        # measurable speedup) and the pipeline has no parallel-decode gather.
        enable_tiling=True,
        enable_slicing=True,
    )

    def _calculate_hybrid_attention_step_multiplier(self, input_args: dict) -> int:
        # The per-step schedule is advanced once per transformer forward
        # (see increment_step_counter in transformer_ideogram4). Ideogram4 runs a
        # separate conditional and unconditional transformer, so the number of
        # forwards per denoising step -- and therefore the schedule length -- depends
        # on the classifier-free-guidance layout:
        #   * guidance on, no CFG-parallel: one process runs both forwards -> 2
        #   * CFG-parallel: the two forwards are split across ranks, one per rank -> 1
        #   * no guidance: single forward -> 1
        guidance_scale = input_args.get("guidance_scale")
        do_cfg = guidance_scale is not None and guidance_scale > 1.0
        if do_cfg and not self.config.use_cfg_parallel:
            return 2
        return 1
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
        # Dequantize FP8 weights to BF16 nn.Linear so torchao/AITER
        # quantization can operate on standard Linear layers.
        state_dict = _dequantize_fp8_state_dict(state_dict, dtype=torch.bfloat16)
        state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
        model.to(device=device, dtype=torch.bfloat16)
        model.load_state_dict(state_dict, strict=False, assign=True)
        model.eval()
        log(f"Loaded FP8 transformer (dequantized to BF16) from {model_id}/{subfolder}")
        return model

    def _load_fp8_text_encoder(self, model_id, device):
        """Load FP8 text encoder, dequantizing to BF16."""
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(
            model_id, subfolder="text_encoder", trust_remote_code=True
        )
        model = AutoModel.from_config(config, trust_remote_code=True)
        state_dict = _load_sharded_safetensors(model_id, "text_encoder", basename="model")
        state_dict = _dequantize_fp8_state_dict(state_dict, dtype=torch.bfloat16)
        state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
        model.to(device=device, dtype=torch.bfloat16)
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        if unexpected:
            log(f"Warning: unexpected keys in text encoder: {unexpected[:5]}")
        model.eval()
        log(f"Loaded FP8 text encoder (dequantized to BF16) from {model_id}/text_encoder")
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
        # Pre-upsample prompt before compile so all warmup/timed runs use
        # the same caption length, preventing recompilation from shape changes.
        prompt = input_args["prompt"]
        if not self._is_json_prompt(prompt) and hasattr(self.pipe, "prompt_enhancer_head") and self.pipe.prompt_enhancer_head is not None:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            if rank == 0:
                caption = self.pipe.upsample_prompt(
                    prompt, height=input_args["height"], width=input_args["width"],
                    device=self.pipe._execution_device,
                )
                if isinstance(caption, list):
                    caption = caption[0]
            else:
                caption = None
            if world_size > 1:
                caption_list = [caption]
                dist.broadcast_object_list(caption_list, src=0)
                caption = caption_list[0]
            input_args["prompt"] = caption
            log(f"Pre-upsampled prompt for compile ({len(caption)} chars)")

        # Pre-set sequence layout before compile to avoid .item() graph breaks.
        # The conditional transformer sees [pad|text|image], the unconditional
        # sees [image] only. Compute the actual text token count from the
        # upsampled prompt to get the right layout.
        height = input_args["height"]
        width = input_args["width"]
        grid_h = height // (self.pipe.vae_scale_factor * self.pipe.patch_size)
        grid_w = width // (self.pipe.vae_scale_factor * self.pipe.patch_size)
        num_image_tokens = grid_h * grid_w
        max_seq = 2048  # default max_sequence_length

        prompt = input_args["prompt"]
        if hasattr(self.pipe, 'tokenizer') and prompt:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = self.pipe.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            num_text = len(self.pipe.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0])
        else:
            num_text = 50
        num_pad = max_seq - num_text

        if hasattr(self.pipe.transformer, '_set_sequence_layout'):
            self.pipe.transformer._set_sequence_layout(num_pad, num_text, num_image_tokens)
        if hasattr(self.pipe.unconditional_transformer, '_set_sequence_layout'):
            self.pipe.unconditional_transformer._set_sequence_layout(0, 0, num_image_tokens)

        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")
        self.pipe.unconditional_transformer = torch.compile(
            self.pipe.unconditional_transformer, mode="default"
        )
        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = 2
        self._run_timed_pipe(compile_args)
