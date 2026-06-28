# Ideogram 4 — xDiT Parallel Inference Results

## Model

- **Model:** [`ideogram-ai/ideogram-4-fp8`](https://huggingface.co/ideogram-ai/ideogram-4-fp8) (9.3B, official FP8 weights)
- **Architecture:** Single-stream DiT, 34 layers, 18 heads, head_dim=256, hidden_size=4608
- **Text encoder:** Qwen3-VL-8B (taps 13 intermediate layers, 53248-dim features)
- **VAE:** AutoencoderKLFlux2 (same as FLUX.2)
- **Prompt enhancer:** `diffusers/qwen3-vl-8b-instruct-lm-head` (auto JSON caption conversion)

## Hardware & Software

- **GPUs:** AMD Instinct MI355X (288GB VRAM per GPU)
- **ROCm:** 6.4
- **Base image:** `amdsiloai/pytorch-xdit-custom:v26.6-20260611`
- **FP8 attention image:** `amdsiloai/pytorch-xdit-custom:v26.6-20260611-aiter-hd256` (built from base + [ROCm/aiter PR #3732](https://github.com/ROCm/aiter/pull/3732) for HD256 FP8 FMHA ASM kernels on gfx950)
- **Attention backends:** AITER MHA CK-tile for BF16 hd256; AITER ASM FP8 hd256 for `--attention_backend AITER_FP8`
- **diffusers:** 0.39.0.dev0 (from `main` branch, required for `Ideogram4Pipeline`)

### Reproducing the FP8 attention image

```bash
# From the base image, rebuild AITER with the HD256 FP8 PR:
docker run -d --name ideogram4 --device=/dev/kfd --device=/dev/dri \
  --group-add video --shm-size=256g \
  -v /hf_cache:/hf_cache \
  amdsiloai/pytorch-xdit-custom:v26.6-20260611 sleep infinity

docker exec ideogram4 bash -c '
  pip install "git+https://github.com/huggingface/diffusers.git" --no-deps
  pip uninstall aiter -y
  cd /app/external/aiter
  rm -rf aiter/jit/build/ aiter/jit/*.so build/ dist/ *.egg-info/
  git fetch origin pull/3732/head:pr-3732
  git checkout pr-3732
  pip install -e . --no-build-isolation
'
```

## Performance

2048x2048, 48 steps, `guidance_scale=7.0`, `torch.compile(mode="default")`, 2 warmup + 2 timed runs.

| Config | 1 GPU | Speedup | USP u=2 (2 GPU) | Speedup | CFG=2 (2 GPU) | Speedup |
|--------|:-----:|:-------:|:---------------:|:-------:|:-------------:|:-------:|
| BF16 GEMMs | 60.2s | 1.00x | 41.1s | 1.47x | 35.4s | 1.70x |
| FP8 GEMMs | 52.6s | 1.14x | 36.1s | 1.67x | 31.1s | 1.94x |
| FP4 GEMMs | 48.3s | 1.25x | 34.5s | 1.74x | 28.9s | 2.08x |
| FP8 attn | 39.6s | 1.52x | 19.3s | 3.12x | 22.4s | 2.69x |
| FP8 attn + FP8 GEMMs | 31.6s | 1.91x | 15.0s | 4.01x | 18.8s | 3.20x |
| **FP8 attn + FP4 GEMMs** | **26.4s** | **2.28x** | **12.2s** | **4.94x** | **16.3s** | **3.69x** |

### Key Observations

- **FP8 attention** (AITER ASM hd256 kernel) provides 1.52x on its own and dramatically improves USP scaling — USP u=2 goes from 1.47x to 3.12x with FP8 attention because the faster attention reduces the relative cost of the all-to-all communication.
- **FP4 GEMMs** (AITER `gemm_a4w4` with fused HIP activation quantization) consistently add ~20% on top of any config, without requiring `torch.compile`.
- **FP8 GEMMs** (torchao `Float8DynamicActivation`) require `torch.compile` to fuse the per-tensor dynamic quantization ops. Without compile, they show no speedup.
- **USP u=2 outperforms CFG=2** when FP8 attention is enabled (12.2s vs 16.3s) because USP parallelizes both transformers' attention, while CFG only eliminates one sequential transformer call.
- **Best 2-GPU config:** USP u=2 + FP8 attention + FP4 GEMMs = **12.2s (4.94x)**

## Usage

### Basic (1 GPU)

```bash
xdit --model ideogram-ai/ideogram-4-fp8 \
  --height 2048 --width 2048 \
  --num_inference_steps 48 \
  --guidance_scale 7.0 \
  --use_torch_compile \
  --prompt "A photo of a cat holding a sign that says hello world"
```

### With FP4 GEMMs

```bash
xdit --model ideogram-ai/ideogram-4-fp8 \
  --use_torch_compile --use_fp4_gemms \
  --height 2048 --width 2048 \
  --prompt "A photo of a cat holding a sign that says hello world"
```

### With FP8 Attention + FP4 GEMMs (fastest single GPU)

Requires AITER with [PR #3732](https://github.com/ROCm/aiter/pull/3732) for HD256 FP8 FMHA.

```bash
xdit --model ideogram-ai/ideogram-4-fp8 \
  --use_torch_compile --use_fp4_gemms \
  --attention_backend AITER_FP8 \
  --height 2048 --width 2048 \
  --prompt "A photo of a cat holding a sign that says hello world"
```

### USP u=2 + FP8 Attention + FP4 GEMMs (fastest 2 GPU)

```bash
torchrun --nproc_per_node=2 -m xfuser.runner \
  --model ideogram-ai/ideogram-4-fp8 \
  --ulysses_degree 2 \
  --use_torch_compile --use_fp4_gemms \
  --attention_backend AITER_FP8 \
  --height 2048 --width 2048 \
  --prompt "A photo of a cat holding a sign that says hello world"
```

### CFG Parallel (2 GPU)

```bash
torchrun --nproc_per_node=2 -m xfuser.runner \
  --model ideogram-ai/ideogram-4-fp8 \
  --use_cfg_parallel \
  --use_torch_compile --use_fp4_gemms \
  --attention_backend AITER_FP8 \
  --height 2048 --width 2048 \
  --prompt "A photo of a cat holding a sign that says hello world"
```

## Notes

- **Prompt format:** Ideogram 4 was trained on structured JSON captions. Plain text prompts produce poor results. The prompt enhancer (`diffusers/qwen3-vl-8b-instruct-lm-head`) automatically converts plain text to JSON captions. JSON prompts are passed through unchanged.
- **FP8 weight loading:** The official `ideogram-ai/ideogram-4-fp8` uses the ideogram-oss native FP8 format (fused QKV). The runner includes a key conversion layer that maps to diffusers format (separate Q/K/V) and dequantizes to BF16 `nn.Linear` for compatibility with torchao/AITER GEMM quantization.
- **HuggingFace authentication:** The model is gated. Set `HF_TOKEN` or run `huggingface-cli login`.
- **torch.compile:** Required for FP8 GEMMs to show speedup. Recommended for all configs. First run includes compilation overhead (~2-3 min warmup).

## Sample Images

Generated images for all configurations at `sweep1_images/`:

```
sweep1_images/1gpu_bf16.png          # 1 GPU, BF16
sweep1_images/1gpu_fp8.png           # 1 GPU, FP8 GEMMs
sweep1_images/1gpu_fp4.png           # 1 GPU, FP4 GEMMs
sweep1_images/1gpu_fp8attn.png       # 1 GPU, FP8 attention
sweep1_images/1gpu_fp8attn_fp8.png   # 1 GPU, FP8 attn + FP8 GEMMs
sweep1_images/1gpu_fp8attn_fp4.png   # 1 GPU, FP8 attn + FP4 GEMMs
sweep1_images/u2_bf16.png            # USP u=2, BF16
sweep1_images/u2_fp8.png             # USP u=2, FP8 GEMMs
sweep1_images/u2_fp4.png             # USP u=2, FP4 GEMMs
sweep1_images/u2_fp8attn.png         # USP u=2, FP8 attention
sweep1_images/u2_fp8attn_fp8.png     # USP u=2, FP8 attn + FP8 GEMMs
sweep1_images/u2_fp8attn_fp4.png     # USP u=2, FP8 attn + FP4 GEMMs
sweep1_images/cfg_bf16.png           # CFG=2, BF16
sweep1_images/cfg_fp8.png            # CFG=2, FP8 GEMMs
sweep1_images/cfg_fp4.png            # CFG=2, FP4 GEMMs
sweep1_images/cfg_fp8attn.png        # CFG=2, FP8 attention
sweep1_images/cfg_fp8attn_fp8.png    # CFG=2, FP8 attn + FP8 GEMMs
sweep1_images/cfg_fp8attn_fp4.png    # CFG=2, FP8 attn + FP4 GEMMs
```

## Files

| File | Description |
|------|-------------|
| `xfuser/model_executor/models/runner_models/ideogram4.py` | Runner model: FP8 loading, key conversion, prompt enhancer, compile |
| `xfuser/model_executor/models/transformers/transformer_ideogram4.py` | Transformer wrapper: USP attention, FP8 attention, SP chunking |
| `xfuser/model_executor/pipelines/pipeline_ideogram4.py` | Pipeline wrapper: CFG parallelism, prompt broadcast |
