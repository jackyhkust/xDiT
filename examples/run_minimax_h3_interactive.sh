#!/bin/bash
# Interactive, continuous MiniMax-H3 I2V stream, playable in VLC. Run INSIDE the
# xDiT container (see examples/docker_minimax_h3_interactive.sh for the host side).
#
#   bash examples/run_minimax_h3_interactive.sh [--demo]
#
# By default the stream opens on assets/amd_gpu_keynote.jpg (a presenter holding
# up an AMD GPU on a keynote stage) with the prompt in assets/amd_gpu_keynote.txt.
# Override with ANCHOR=/path/anchor.png PROMPT="..." (or PROMPT_FILE=...).
#
# Then, in another shell in the same container:
#   python -m xfuser.streaming.prompt_cli --stream_dir /dev/shm/h3_stream
#   # type a new scene; it transitions in smoothly without a cut.
#   # type /overlay to show or hide the buffered frames and latest input on the video.
#
# Demo mode (pass --demo or set DEMO=1): after DEMO_INTERVAL_S (30) seconds with
# no new prompt, the console queues a scripted one: a character joins (or, at
# three, one leaves and another joins), the cast promotes an AMD product, with a
# random storyline and style. Typed prompts still work and restart the countdown.
# DEMO_CAST="the white cat" counts characters already in the opening scene.
#
# Watch it (from a laptop first open a tunnel to the node):
#   ssh -N -L 18080:127.0.0.1:20123 yatholam@smci355-ccs-aus-m06-17.cs-aus.dcgpu
#   vlc --network-caching=500 http://127.0.0.1:18080/stream.ts   # stats at /stats
set -euo pipefail

for arg in "$@"; do
    case "$arg" in
        --demo) DEMO=1 ;;
        *) echo "Unknown argument: $arg (settings are environment variables)" >&2; exit 2 ;;
    esac
done

REPO=$(cd "$(dirname "$0")/.." && pwd)
DEFAULT_ANCHOR="$REPO/assets/amd_gpu_keynote.jpg"
if [[ -z "${ANCHOR:-}" ]]; then
    ANCHOR=$DEFAULT_ANCHOR
    PROMPT_FILE=${PROMPT_FILE:-$REPO/assets/amd_gpu_keynote.txt}
    # The presenter is already on stage; the demo counts her toward its three characters.
    DEMO_CAST=${DEMO_CAST-the presenter in the lavender blazer}
fi
if [[ -z "${PROMPT:-}" ]]; then
    [[ -n "${PROMPT_FILE:-}" ]] || { echo "Set PROMPT or PROMPT_FILE for ANCHOR=$ANCHOR." >&2; exit 2; }
    PROMPT=$(<"$PROMPT_FILE")
fi
# Canvas: follows the anchor's aspect ratio at about AREA pixels unless WIDTH and
# HEIGHT are both given (e.g. a 832x1104 portrait anchor -> 640x832).
AREA=${AREA:-$((960 * 544))}
if [[ -z "${WIDTH:-}" || -z "${HEIGHT:-}" ]]; then
    read -r WIDTH HEIGHT < <(python "$(dirname "$0")/../xfuser/streaming/resolution.py" "$ANCHOR" --area "$AREA")
fi
NUM_FRAMES=${NUM_FRAMES:-124}
STEPS=${STEPS:-12}
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
PROMPT_ENHANCER=${PROMPT_ENHANCER:-qwen3vl}
ENHANCER_MAX_NEW_TOKENS=${ENHANCER_MAX_NEW_TOKENS:-384}
CRF=${CRF:-23}
MAX_KBPS=${MAX_KBPS:-3000}
COND_COLOR_MATCH=${COND_COLOR_MATCH:-none}
INTERACTIVE_CLI=${INTERACTIVE_CLI:-1}
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
    > "$OUTPUT_DIR/streamer.log" 2>&1 &
STREAMER_PID=$!

# 2. Generator: torchrun job that chains I2V chunks and applies viewer prompts.
torchrun --nproc_per_node "$NPROC" entrypoints/minimax_h3_stream.py \
    --model MiniMax-H3 --task i2va \
    --prompt "$PROMPT" --input_images "$ANCHOR" \
    --height "$HEIGHT" --width "$WIDTH" --num_frames "$NUM_FRAMES" --num_inference_steps "$STEPS" \
    --attention_backend aiter_bf16 --use_fp8_gemms \
    --ulysses_degree "$NPROC" --use_parallel_vae --use_torch_compile --warmup_calls 1 \
    --stream_dir "$STREAM_DIR" --duration_s "$DURATION_S" --output_directory "$OUTPUT_DIR" \
    --max_ahead_chunks "$MAX_AHEAD_CHUNKS" --transition_chunks "$TRANSITION_CHUNKS" \
    --prompt_enhancer "$PROMPT_ENHANCER" --enhancer_max_new_tokens "$ENHANCER_MAX_NEW_TOKENS" \
    --cond_color_match "$COND_COLOR_MATCH" "$CUT_IN_FLAG" \
    $EXTRA_ARGS > "$OUTPUT_DIR/generator.log" 2>&1 &
GENERATOR_PID=$!

DEMO_PID=""
cleanup() { kill "$STREAMER_PID" "$GENERATOR_PID" $DEMO_PID 2>/dev/null || true; }
trap cleanup INT TERM EXIT

echo "Canvas ${WIDTH}x${HEIGHT}. Streamer PID $STREAMER_PID, generator PID $GENERATOR_PID."
echo "Logs: $OUTPUT_DIR/{streamer,generator}.log"
echo "Stream: http://127.0.0.1:$PORT/stream.ts  (stats at /stats)"

if [[ "$INTERACTIVE_CLI" == "1" ]]; then
    # Foreground prompt console; closing it (Ctrl-D) stops the stream via the trap.
    echo "Type new prompts below. /overlay toggles the on-video info. Ctrl-D to stop everything."
    python -m xfuser.streaming.prompt_cli --stream_dir "$STREAM_DIR" "${DEMO_ARGS[@]}" || true
else
    # Headless: run the demo loop (if enabled) and wait for the generator to finish.
    if [[ "$DEMO" == "1" ]]; then
        python -m xfuser.streaming.prompt_cli --stream_dir "$STREAM_DIR" "${DEMO_ARGS[@]}" --no_console \
            > "$OUTPUT_DIR/demo.log" 2>&1 &
        DEMO_PID=$!
    fi
    wait "$GENERATOR_PID"
fi
