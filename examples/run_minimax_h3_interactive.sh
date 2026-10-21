#!/bin/bash
# Interactive, continuous MiniMax-H3 I2V stream, watched and steered from a web
# page (or played in VLC). Run INSIDE the xDiT container (see
# examples/docker_minimax_h3_interactive.sh for the host side).
#
#   bash examples/run_minimax_h3_interactive.sh [--cli] [--demo]
#
# By default the stream opens on assets/amd_ai_keynote.jpg (a glowing particle
# wave on a keynote LED wall) with the prompt in assets/amd_ai_keynote.txt (a
# white-haired Asian woman walks on and promotes AMD GPUs and the data center).
# Override with ANCHOR=/path/anchor.png PROMPT="..." (or PROMPT_FILE=...).
#
# Prompts come from the web page; the terminal prints them as they are queued.
# Ctrl-C stops everything.
#
# --cli (or CLI=1) also opens the terminal prompt console: type a new scene and
# it transitions in smoothly; /overlay shows or hides the buffered frames and
# latest input on the video; /status prints the prompt state. /quit or Ctrl-D
# closes only the console; the stream, page and demo keep running. More consoles
# can be opened in other shells in the same container:
#   python -m xfuser.streaming.prompt_cli --stream_dir /dev/shm/h3_stream
#
# Demo mode (pass --demo or set DEMO=1): after DEMO_INTERVAL_S (30) seconds with
# no new prompt from anywhere (page, console or demo), a scripted one is queued:
# a character joins (or, at three, one leaves and another joins), the cast
# promotes an AMD product, with a random storyline and style. It works with or
# without --cli. DEMO_CAST="the white cat" counts characters already in the
# opening scene.
#
# FAKE_GENERATOR=1 swaps the model for a CPU test pattern (no GPUs, starts in
# seconds) to try the page, prompts, overlay and demo mode.
#
# MODEL=ltx25 streams LTX-2.5 distilled instead of MiniMax-H3, on a smaller
# 768x448 canvas (AREA, WIDTH/HEIGHT override it; both must be multiples of 64)
# with the prompt in assets/amd_ai_keynote_ltx.txt. Viewer prompts are rewritten
# by LTX-2.5's bundled Gemma 4 prompt enhancer from the frame on screen.
# PROMPT_ENHANCER=none uses the raw text instead; it then replaces the opening
# prompt (PROMPT_FALLBACK=append adds it to the opening prompt). NPROC=1 runs it
# on one GPU.
#
# Watch it (from a laptop first open a tunnel to the node):
#   ssh -N -L 18080:127.0.0.1:20123 yatholam@smci355-ccs-aus-m06-17.cs-aus.dcgpu
#   open http://127.0.0.1:18080/ in Chrome: video, prompt box, overlay switch, status
#   or: vlc --network-caching=500 http://127.0.0.1:18080/stream.ts   # stats at /stats
set -euo pipefail

for arg in "$@"; do
    case "$arg" in
        --demo) DEMO=1 ;;
        --cli) CLI=1 ;;
        *) echo "Unknown argument: $arg (settings are environment variables)" >&2; exit 2 ;;
    esac
done

REPO=$(cd "$(dirname "$0")/.." && pwd)
# MODEL=h3 (default) streams MiniMax-H3 I2VA. MODEL=ltx25 streams LTX-2.5
# distilled: 8 fixed steps, 121-frame chunks, 48 kHz audio, a smaller default
# canvas (768x448) so the generator runs well ahead of playback, and the
# plain-text prompt fallback (it has no Qwen3-VL conditioner).
MODEL=${MODEL:-h3}
case "$MODEL" in
    h3)
        MODEL_ID=MiniMax-H3; TASK=i2va; DEFAULT_PROMPT_NAME=amd_ai_keynote.txt
        DEFAULT_AREA=$((960 * 544)); DIVISOR=32; DEFAULT_NUM_FRAMES=124; DEFAULT_STEPS=12
        DEFAULT_ENHANCER=qwen3vl; DEFAULT_FALLBACK=append; ENHANCE_ESTIMATE_S=20; SAMPLE_RATE=32000
        ;;
    ltx25)
        [[ -z "${TURBO:-}" ]] || { echo "TURBO is a MiniMax-H3 LoRA; unset it for MODEL=ltx25." >&2; exit 2; }
        MODEL_ID=LTX-2.5-distilled; TASK=i2v; DEFAULT_PROMPT_NAME=amd_ai_keynote_ltx.txt
        DEFAULT_AREA=$((768 * 448)); DIVISOR=64; DEFAULT_NUM_FRAMES=121; DEFAULT_STEPS=8
        DEFAULT_ENHANCER=gemma4; DEFAULT_FALLBACK=replace; ENHANCE_ESTIMATE_S=8; SAMPLE_RATE=48000
        ;;
    *) echo "Unknown MODEL=$MODEL (use h3 or ltx25)." >&2; exit 2 ;;
esac

DEFAULT_ANCHOR="$REPO/assets/amd_ai_keynote.jpg"
if [[ -z "${ANCHOR:-}" ]]; then
    ANCHOR=$DEFAULT_ANCHOR
    PROMPT_FILE=${PROMPT_FILE:-$REPO/assets/$DEFAULT_PROMPT_NAME}
    # The presenter walks on in the opening shot; the demo counts her toward its three characters.
    DEMO_CAST=${DEMO_CAST-the white-haired presenter in the dark blazer}
fi
if [[ -z "${PROMPT:-}" ]]; then
    [[ -n "${PROMPT_FILE:-}" ]] || { echo "Set PROMPT or PROMPT_FILE for ANCHOR=$ANCHOR." >&2; exit 2; }
    PROMPT=$(<"$PROMPT_FILE")
fi
# Canvas: follows the anchor's aspect ratio at about AREA pixels unless WIDTH and
# HEIGHT are both given (e.g. a 832x1104 portrait anchor -> 640x832).
AREA=${AREA:-$DEFAULT_AREA}
if [[ -z "${WIDTH:-}" || -z "${HEIGHT:-}" ]]; then
    read -r WIDTH HEIGHT < <(python "$(dirname "$0")/../xfuser/streaming/resolution.py" "$ANCHOR" \
        --area "$AREA" --divisor "$DIVISOR")
fi
NUM_FRAMES=${NUM_FRAMES:-$DEFAULT_NUM_FRAMES}
# TURBO: distilled lightx2v Turbo LoRA for few-step sampling. TURBO=8 (or 4)
# merges the matching adapter and drops STEPS to that count; TURBO=/path.safetensors
# uses a specific file. H3_TURBO_LORA is read by the model loader (minimax_h3.py).
TURBO_LORA_DIR=${TURBO_LORA_DIR:-/turbo_lora}
case "${TURBO:-}" in
    "") ;;
    8) export H3_TURBO_LORA="$TURBO_LORA_DIR/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"; STEPS=${STEPS:-8} ;;
    4) export H3_TURBO_LORA="$TURBO_LORA_DIR/minimax_h3_fl2v_turbo_4step_v0.1.safetensors"; STEPS=${STEPS:-4} ;;
    *) export H3_TURBO_LORA="$TURBO" ;;
esac
STEPS=${STEPS:-$DEFAULT_STEPS}
# Effectively unlimited: ~100 hours. The stream runs until you stop it.
DURATION_S=${DURATION_S:-360000}
NPROC=${NPROC:-8}
PORT=${PORT:-20123}
STREAM_DIR=${STREAM_DIR:-/dev/shm/h3_stream}
OUTPUT_DIR=${OUTPUT_DIR:-/output/stream}
# Buffer cap. With cut-in it only has to outlast the enhancement pause plus one
# chunk: ceil((max enhance s + 4 s lead + 3 s margin) / 5.125). 4 suits the
# 384-token enhancer; use 6 for ENHANCER_MAX_NEW_TOKENS=768, 2-3 with CUT_IN=0.
MAX_AHEAD_CHUNKS=${MAX_AHEAD_CHUNKS:-4}
TRANSITION_CHUNKS=${TRANSITION_CHUNKS:-1}
PROMPT_ENHANCER=${PROMPT_ENHANCER:-$DEFAULT_ENHANCER}
PROMPT_FALLBACK=${PROMPT_FALLBACK:-$DEFAULT_FALLBACK}
ENHANCER_MAX_NEW_TOKENS=${ENHANCER_MAX_NEW_TOKENS:-384}
CRF=${CRF:-23}
MAX_KBPS=${MAX_KBPS:-3000}
COND_COLOR_MATCH=${COND_COLOR_MATCH:-none}
CLI=${CLI:-${INTERACTIVE_CLI:-0}}
FAKE_GENERATOR=${FAKE_GENERATOR:-0}
# Initial overlay state; /overlay in the prompt console switches it at runtime.
OVERLAY=${OVERLAY:-0}
CUT_IN=${CUT_IN:-1}
DEMO=${DEMO:-0}
DEMO_INTERVAL_S=${DEMO_INTERVAL_S:-30}
DEMO_CAST=${DEMO_CAST:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
DEMO_ARGS=()
if [[ "$DEMO" == "1" ]]; then
    DEMO_ARGS=(--demo --demo_interval_s "$DEMO_INTERVAL_S" --demo_cast "$DEMO_CAST")
fi
CUT_IN_FLAG=$([[ "$CUT_IN" == "1" ]] && echo --cut_in || echo --no-cut_in)
OVERLAY_FLAG=$([[ "$OVERLAY" == "1" ]] && echo --overlay || echo --no-overlay)

mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")/.."

# 1. Streamer: plays chunks out at 24 fps (optional status overlay), records the TS.
python -m xfuser.streaming.ts_server \
    --stream_dir "$STREAM_DIR" --port "$PORT" --width "$WIDTH" --height "$HEIGHT" \
    --anchor_image "$ANCHOR" --metrics_path "$OUTPUT_DIR/metrics.jsonl" \
    --record "$OUTPUT_DIR/stream.ts" "$OVERLAY_FLAG" --crf "$CRF" --max_kbps "$MAX_KBPS" \
    --sample_rate "$SAMPLE_RATE" > "$OUTPUT_DIR/streamer.log" 2>&1 &
STREAMER_PID=$!

# 2. Generator: torchrun job that chains I2V chunks and applies viewer prompts.
if [[ "$FAKE_GENERATOR" == "1" ]]; then
    # The streamer clears the stream directory before it listens; start writing after that.
    for _ in $(seq 240); do
        (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && break
        sleep 0.5
    done
    python -m xfuser.streaming.fake_generator \
        --stream_dir "$STREAM_DIR" --width "$WIDTH" --height "$HEIGHT" --num_frames "$NUM_FRAMES" \
        --sample_rate "$SAMPLE_RATE" --anchor "$ANCHOR" --prompt "$PROMPT" --max_ahead_chunks "${FAKE_MAX_AHEAD_CHUNKS:-1}" \
        > "$OUTPUT_DIR/generator.log" 2>&1 &
else
    # Parallel VAE decode splits the decode across ranks; on one rank it only adds tiling.
    PARALLEL_VAE_FLAG=$([[ "$NPROC" -gt 1 ]] && echo --use_parallel_vae || true)
    torchrun --nproc_per_node "$NPROC" entrypoints/minimax_h3_stream.py \
        --model "$MODEL_ID" --task "$TASK" \
        --prompt "$PROMPT" --input_images "$ANCHOR" \
        --height "$HEIGHT" --width "$WIDTH" --num_frames "$NUM_FRAMES" --num_inference_steps "$STEPS" \
        --attention_backend aiter_bf16 --use_fp8_gemms \
        --ulysses_degree "$NPROC" $PARALLEL_VAE_FLAG --use_torch_compile --warmup_calls 1 \
        --stream_dir "$STREAM_DIR" --duration_s "$DURATION_S" --output_directory "$OUTPUT_DIR" \
        --max_ahead_chunks "$MAX_AHEAD_CHUNKS" --transition_chunks "$TRANSITION_CHUNKS" \
        --prompt_enhancer "$PROMPT_ENHANCER" --prompt_fallback "$PROMPT_FALLBACK" \
        --enhance_estimate_s "$ENHANCE_ESTIMATE_S" \
        --enhancer_max_new_tokens "$ENHANCER_MAX_NEW_TOKENS" \
        --cond_color_match "$COND_COLOR_MATCH" "$CUT_IN_FLAG" \
        $EXTRA_ARGS > "$OUTPUT_DIR/generator.log" 2>&1 &
fi
GENERATOR_PID=$!

DEMO_PID=""
FEED_PID=""
cleanup() { kill "$STREAMER_PID" "$GENERATOR_PID" $DEMO_PID $FEED_PID 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "Canvas ${WIDTH}x${HEIGHT}. Streamer PID $STREAMER_PID, generator PID $GENERATOR_PID."
echo "Logs: $OUTPUT_DIR/{streamer,generator,demo,console}.log"
echo "Web page: http://127.0.0.1:$PORT/   VLC: http://127.0.0.1:$PORT/stream.ts  (stats at /stats)"

# 3. Demo loop: its own process, so it keeps going whether or not a console is open.
if [[ "$DEMO" == "1" ]]; then
    python -m xfuser.streaming.prompt_cli --stream_dir "$STREAM_DIR" "${DEMO_ARGS[@]}" --no_console --quiet \
        > "$OUTPUT_DIR/demo.log" 2>&1 &
    DEMO_PID=$!
fi

# 4. Optional terminal console; it also prints prompts from the page and the demo.
if [[ "$CLI" == "1" ]]; then
    echo "Type new prompts below. /overlay toggles the on-video info, /status shows the prompt state."
    echo "/quit or Ctrl-D closes the console (the stream keeps running); Ctrl-C stops everything."
    python -m xfuser.streaming.prompt_cli --stream_dir "$STREAM_DIR" 2>>"$OUTPUT_DIR/console.log" || true
fi

# 5. Until the generator exits: print prompts as they are queued.
echo "Prompts from the web page${DEMO_PID:+ and the demo} are shown below. Ctrl-C stops everything."
python -m xfuser.streaming.prompt_cli --stream_dir "$STREAM_DIR" --no_console 2>>"$OUTPUT_DIR/console.log" &
FEED_PID=$!
wait "$GENERATOR_PID"
