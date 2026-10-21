#!/bin/bash
# Host-side launcher for the interactive MiniMax-H3 stream. Runs the idle check,
# starts the container (named with "yatholam"), and runs the in-container launcher,
# which starts the streamer + generator. Prompts come from the web page; the
# terminal prints them as they arrive. Ctrl-C stops everything.
#
#   bash examples/docker_minimax_h3_interactive.sh [--cli] [--demo]
#
# --cli (or CLI=1) also opens the terminal prompt console (type prompts, /overlay,
# /status); /quit closes only the console, the stream keeps running.
#   GPUS=0,1,2,3,4,5,6,7 ANCHOR=/workspace/xDiT/assets/i2v_input.JPG \
#     PROMPT="a white cat on a surfboard at a sunny beach" bash examples/docker_minimax_h3_interactive.sh
#
# Watch from a laptop:
#   ssh -N -L 18080:127.0.0.1:20123 yatholam@smci355-ccs-aus-m06-17.cs-aus.dcgpu
#   open http://127.0.0.1:18080/ in Chrome: video, prompt box, overlay switch, status
#   or: vlc --network-caching=500 http://127.0.0.1:18080/stream.ts
# If the player still lags behind live, lower the bitrate further: CRF=26 MAX_KBPS=2000.
# The page accepts prompts from anyone who can reach the port; it is bound to
# 127.0.0.1 on the node, so only SSH-tunnel users can.
#
# Add --demo (or DEMO=1) for demo mode: a scripted AMD-promo prompt every
# DEMO_INTERVAL_S (30) seconds without a new prompt from the page or console.
# FAKE_GENERATOR=1 replaces the model with a CPU test pattern to try the page quickly.
#
# MODEL=ltx25 streams LTX-2.5 distilled on a smaller 768x448 canvas instead of
# MiniMax-H3, e.g. on one GPU: GPUS=0 MODEL=ltx25 bash examples/docker_minimax_h3_interactive.sh
#
# TURBO=8 (or TURBO=4) merges the distilled lightx2v Turbo LoRA and samples in
# 8 (or 4) steps instead of 12, for a faster generator. The adapters must live in
# TURBO_LORA_HOST (default ~/phantom/turbo_lora); it is mounted read-only.
set -euo pipefail

for arg in "$@"; do
    case "$arg" in
        --demo) DEMO=1 ;;
        --cli) CLI=1 ;;
        *) echo "Unknown argument: $arg (settings are environment variables)" >&2; exit 2 ;;
    esac
done

IMAGE=${IMAGE:-amdsiloai/pytorch-xdit:v26.10.1}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PORT=${PORT:-20123}
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
HF_ROOT=${HF_ROOT:-$(readlink -f /raid/huggingface)}
# ANCHOR (a path inside the container) and PROMPT default to the AMD AI keynote
# scene in assets/amd_ai_keynote.{jpg,txt}; set both to stream something else.
ANCHOR=${ANCHOR:-}
PROMPT=${PROMPT:-}
NAME=${NAME:-yatholam-xdit-stream-runner-$(date +%Y%m%d)}
REPLACE=${REPLACE:-0}
DRY_RUN=${DRY_RUN:-0}
OUT_DIR=${OUT_DIR:-$HOME/phantom/h3_stream_out}
mkdir -p "$OUT_DIR"
# Distilled lightx2v Turbo LoRA (few-step sampling). TURBO=8 or 4 picks the
# matching adapter from TURBO_LORA_HOST (mounted read-only at /turbo_lora); the
# in-container launcher sets H3_TURBO_LORA and drops STEPS to match.
TURBO_LORA_HOST=${TURBO_LORA_HOST:-$HOME/phantom/turbo_lora}

# ---- idle check (CPU <=80%, every GPU <=5% util AND <=20% VRAM) ----
wait_for_idle() {
    local CPU_MAX=${CPU_THRESHOLD:-80} GPU_MAX=${GPU_THRESHOLD:-5} VRAM_MAX=${VRAM_THRESHOLD:-20} POLL=${POLL_INTERVAL:-30}
    [[ -n "${1:-}" ]] && echo "=== Idle check: $1 ==="
    while true; do
        if command -v vmstat &>/dev/null; then
            CPU=$(vmstat 1 2|tail -1|awk '{printf "%.0f",100-$15}')
        else
            S1=$(head -1 /proc/stat);sleep 1;S2=$(head -1 /proc/stat)
            CPU=$(printf '%s\n' "$S1" "$S2"|awk 'NR==1{for(i=2;i<=NF;i++)t1+=$i;id1=$5}NR==2{for(i=2;i<=NF;i++)t2+=$i;id2=$5}END{printf "%.0f",100*(1-(id2-id1)/(t2-t1))}')
        fi
        [[ -z "$CPU" ]] && { echo "WARN: cannot read CPU%."; break; }
        LOAD_HIGH=0
        if [[ -r /proc/loadavg ]]; then
            awk -v l="$(awk '{print $1}' /proc/loadavg)" -v n="$(nproc 2>/dev/null||echo 1)" \
                'BEGIN{exit(l>0.9*n)?0:1}' && LOAD_HIGH=1
        fi
        (( CPU <= CPU_MAX )) && (( LOAD_HIGH == 0 )) && { echo "CPU OK (${CPU}%)."; break; }
        echo "$(date +%T) CPU busy ${CPU}%"; sleep "$POLL"
    done
    while true; do
        GPU_UTIL=""; GPU_VRAM=""
        if command -v rocm-smi &>/dev/null; then
            GPU_UTIL=$(rocm-smi --showuse 2>/dev/null | grep "GPU use (%)" | awk -F': ' '{print $NF}')
            GPU_VRAM=$(rocm-smi --showmemuse 2>/dev/null | grep "Memory Allocated (VRAM%)" | awk -F': ' '{print $NF}')
        elif command -v nvidia-smi &>/dev/null; then
            GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
            GPU_VRAM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
                | awk -F', *' '{printf "%d\n", ($1*100)/$2}')
        fi
        if [[ -z "$GPU_UTIL$GPU_VRAM" ]]; then
            echo "GPUs idle (or no GPU tool found)."; break
        fi
        mapfile -t UTIL_ARR <<< "$GPU_UTIL"
        mapfile -t VRAM_ARR <<< "$GPU_VRAM"
        N=${#UTIL_ARR[@]}
        [[ ${#VRAM_ARR[@]} -ne $N ]] && echo "WARN: util/vram GPU-count mismatch (${N} vs ${#VRAM_ARR[@]})"
        BUSY=""
        for ((i=0; i<N; i++)); do
            un=${UTIL_ARR[i]//[!0-9]/}; un=${un:-0}
            vn=${VRAM_ARR[i]//[!0-9]/}; vn=${vn:-0}
            (( un > GPU_MAX )) && BUSY+=" GPU${i}=${un}%util"
            (( vn > VRAM_MAX )) && BUSY+=" GPU${i}=${vn}%vram"
        done
        [[ -z "$BUSY" ]] && { echo "GPUs idle (util <=${GPU_MAX}%, vram <=${VRAM_MAX}%)."; break; }
        echo "$(date +%T) GPUs busy:$BUSY"; sleep "$POLL"
    done
    echo "--- System idle. Ready. ---"
}

# ---- container-name ownership checks (must contain "yatholam") ----
case "$NAME" in
    *yatholam*) : ;;
    *) echo "Refusing: container name '$NAME' must contain 'yatholam'." >&2; exit 1 ;;
esac
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    if [[ "$REPLACE" == "1" ]]; then
        case "$NAME" in
            *yatholam*) echo "Removing existing $NAME"; docker rm -f "$NAME" ;;
            *) echo "Refusing to remove '$NAME' (no yatholam marker)." >&2; exit 1 ;;
        esac
    else
        echo "Container $NAME already exists. Set REPLACE=1 to recreate it." >&2
        exit 1
    fi
fi

HIP_VISIBLE_DEVICES="$GPUS"
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")

RUN=(docker run -it --rm --name "$NAME"
    --device /dev/kfd --device /dev/dri --group-add video
    --security-opt seccomp=unconfined --shm-size 64g
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
    -e HF_HOME=/raid/huggingface -e HF_HUB_OFFLINE=1
    -e PYTHONPATH=/workspace/xDiT
    ${ANCHOR:+-e "ANCHOR=$ANCHOR"} ${PROMPT:+-e "PROMPT=$PROMPT"} ${PROMPT_FILE:+-e "PROMPT_FILE=$PROMPT_FILE"}
    -e NPROC="$NPROC" -e PORT="$PORT"
    -e OVERLAY="${OVERLAY:-0}" -e COND_COLOR_MATCH="${COND_COLOR_MATCH:-anchor_mean}"
    -e CUT_IN="${CUT_IN:-1}" ${AREA:+-e "AREA=$AREA"}
    ${MODEL:+-e "MODEL=$MODEL"} ${NUM_FRAMES:+-e "NUM_FRAMES=$NUM_FRAMES"}
    ${PROMPT_FALLBACK:+-e "PROMPT_FALLBACK=$PROMPT_FALLBACK"} ${PROMPT_ENHANCER:+-e "PROMPT_ENHANCER=$PROMPT_ENHANCER"}
    -e CRF="${CRF:-23}" -e MAX_KBPS="${MAX_KBPS:-3000}"
    -e ENHANCER_MAX_NEW_TOKENS="${ENHANCER_MAX_NEW_TOKENS:-384}"
    -e MAX_AHEAD_CHUNKS="${MAX_AHEAD_CHUNKS:-4}"
    -e DEMO="${DEMO:-0}" -e DEMO_INTERVAL_S="${DEMO_INTERVAL_S:-30}" ${DEMO_CAST+-e "DEMO_CAST=$DEMO_CAST"}
    -e FAKE_GENERATOR="${FAKE_GENERATOR:-0}" -e CLI="${CLI:-0}"
    ${TURBO:+-e "TURBO=$TURBO"} ${STEPS:+-e "STEPS=$STEPS"} -e TURBO_LORA_DIR=/turbo_lora
    ${WIDTH:+-e WIDTH="$WIDTH"} ${HEIGHT:+-e HEIGHT="$HEIGHT"}
    -e OUTPUT_DIR=/output/stream
    -v "$SRC":/workspace/xDiT -v "$HF_ROOT":/raid/huggingface -v "$OUT_DIR":/output/stream
    ${TURBO:+-v "$TURBO_LORA_HOST":/turbo_lora:ro}
    -w /workspace/xDiT
    -p 127.0.0.1:"$PORT":"$PORT"
    "$IMAGE"
    bash examples/run_minimax_h3_interactive.sh)

echo "Container: $NAME"
echo "Mount: $SRC -> /workspace/xDiT ; $HF_ROOT -> /raid/huggingface"
printf 'Command: '; printf '%q ' "${RUN[@]}"; echo

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1, not launching."
    exit 0
fi

wait_for_idle "minimax-h3 interactive stream"
exec "${RUN[@]}"
