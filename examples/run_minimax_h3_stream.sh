#!/bin/bash
# Continuous MiniMax-H3 I2V stream, playable in VLC:
#
#   ANCHOR=/path/anchor.png PROMPT="..." bash examples/run_minimax_h3_stream.sh
#   vlc http://<host>:20123/stream.ts      # from a laptop: ssh -L 20123:localhost:20123 <host>
#
# Each chunk is conditioned on the last frame of the previous one. The streamer
# plays them out at 24 fps; GET http://<host>:20123/stats shows buffer depth,
# stall frames (0 means the stream kept real time) and per-chunk drift metrics.
set -euo pipefail

ANCHOR=${ANCHOR:?set ANCHOR to the first-frame image}
PROMPT=${PROMPT:?set PROMPT}
HEIGHT=${HEIGHT:-544}
WIDTH=${WIDTH:-960}
NUM_FRAMES=${NUM_FRAMES:-124}
STEPS=${STEPS:-12}
DURATION_S=${DURATION_S:-330}
NPROC=${NPROC:-8}
PORT=${PORT:-20123}
STREAM_DIR=${STREAM_DIR:-/dev/shm/h3_stream}
OUTPUT_DIR=${OUTPUT_DIR:-/output/stream}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")/.."

python -m xfuser.streaming.ts_server \
    --stream_dir "$STREAM_DIR" --port "$PORT" --width "$WIDTH" --height "$HEIGHT" \
    --anchor_image "$ANCHOR" --metrics_path "$OUTPUT_DIR/metrics.jsonl" \
    --record "$OUTPUT_DIR/stream.ts" --exit_when_done \
    > "$OUTPUT_DIR/streamer.log" 2>&1 &
STREAMER_PID=$!
trap 'kill $STREAMER_PID 2>/dev/null || true' INT TERM
sleep 2

torchrun --nproc_per_node "$NPROC" entrypoints/minimax_h3_stream.py \
    --model MiniMax-H3 --task i2va \
    --prompt "$PROMPT" --input_images "$ANCHOR" \
    --height "$HEIGHT" --width "$WIDTH" --num_frames "$NUM_FRAMES" --num_inference_steps "$STEPS" \
    --attention_backend aiter_bf16 --use_fp8_gemms \
    --ulysses_degree "$NPROC" --use_parallel_vae --use_torch_compile --warmup_calls 1 \
    --stream_dir "$STREAM_DIR" --duration_s "$DURATION_S" --output_directory "$OUTPUT_DIR" \
    $EXTRA_ARGS 2>&1 | tee "$OUTPUT_DIR/generator.log"

wait $STREAMER_PID
