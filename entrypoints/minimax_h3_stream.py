"""Continuous, interactive MiniMax-H3 I2V generation for a live stream.

Every chunk is an I2V request conditioned on the last frame of the chunk before
it; chunk 0 is conditioned on the ``--input_images`` anchor. The last rank
writes each finished chunk to ``--stream_dir``, where
``python -m xfuser.streaming.ts_server`` picks it up and plays it out. Start the
streamer first: it clears the stream directory on startup.

    torchrun --nproc_per_node 8 entrypoints/minimax_h3_stream.py \\
        --model MiniMax-H3 --task i2va --prompt "..." --input_images anchor.png \\
        --height 544 --width 960 --num_frames 124 --num_inference_steps 12 \\
        --attention_backend aiter_bf16 --use_fp8_gemms --ulysses_degree 8 \\
        --use_parallel_vae --use_torch_compile --stream_dir /dev/shm/h3_stream

LTX-2.5 distilled streams the same way (``--model LTX-2.5-distilled --task i2v
--num_frames 121 --num_inference_steps 8``; 48 kHz audio, so start the streamer
with ``--sample_rate 48000``). Viewer prompts are rewritten by its bundled
Gemma 4 prompt enhancer (``--prompt_enhancer gemma4``) into one caption that
starts from the current frame; with ``--prompt_enhancer none
--prompt_fallback replace`` the raw text replaces the opening prompt instead.

Viewers change the scene at runtime with
``python -m xfuser.streaming.prompt_cli --stream_dir /dev/shm/h3_stream``. Each
new prompt is rewritten by the model's own Qwen3-VL conditioner, using the last
generated frame for continuity, into a smooth TRANSITION prompt followed by a
CONTINUATION prompt, so the scene changes without a cut.

A new prompt does not wait for the whole buffer to drain. The generator
estimates how long the transition needs (prompt enhancement plus one chunk plus
a margin), takes the frame that will be on screen at that moment as the
reference frame, and tells the streamer to drop every buffered frame after it
(``--cut_in``). The transition chunk is conditioned on that reference frame, so
it continues seamlessly from exactly where playback will be.

To stay real-time, a chunk must take no longer to generate than its new frames
take to play: ``(num_frames - 1) / fps`` seconds. Prompt enhancement pauses
generation, so keep enough buffer (``--max_ahead_chunks``) to cover it.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import get_world_group
from xfuser.core.utils.runner_utils import is_last_process, log
from xfuser.runner import setup_logging, xFuserModelRunner
from xfuser.streaming.chunks import Chunk, mark_done, read_position, write_chunk, write_cut
from xfuser.streaming.color import match_color_stats
from xfuser.streaming.enhancer import LtxPromptEnhancer, PromptEnhancer
from xfuser.streaming.history import FrameHistory
from xfuser.streaming.local_snapshot import localize_modular_components, resolve_local_snapshot
from xfuser.streaming.model_io import chunk_media, with_condition
from xfuser.streaming.resolution import cover_resize, upright
from xfuser.streaming.prompt_queue import (
    STATE_ENHANCING,
    STATE_LIVE,
    STATE_RENDERING,
    read_new_prompts,
    write_prompt_state,
)

PHASE_ORIGINAL = "original"
PHASE_TRANSITION = "transition"
PHASE_CONTINUATION = "continuation"


def build_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(description="Continuous MiniMax-H3 I2V generation for streaming")
    xFuserArgs.add_runner_args(parser)
    group = parser.add_argument_group("Streaming")
    group.add_argument("--stream_dir", type=Path, default=Path("/dev/shm/h3_stream"))
    group.add_argument("--duration_s", type=float, default=330.0, help="Stream length to generate, in seconds.")
    group.add_argument(
        "--max_ahead_chunks",
        type=int,
        default=4,
        help=(
            "Pause generation while this many chunks' worth of frames are still waiting to play. "
            "With --cut_in it must outlast the enhancement pause plus one chunk."
        ),
    )
    group.add_argument(
        "--cond_color_match",
        choices=["none", "anchor", "anchor_mean"],
        default="none",
        help=(
            "Pull the conditioning frame's L*a*b* statistics toward the anchor to counter chained drift: "
            "'anchor' matches mean and std, 'anchor_mean' only shifts the mean."
        ),
    )
    group.add_argument("--cond_color_strength", type=float, default=0.5)
    group.add_argument(
        "--prompt_enhancer",
        choices=["qwen3vl", "gemma4", "none"],
        default="qwen3vl",
        help=(
            "How to turn a viewer's prompt into a model prompt using the frame it starts from: 'qwen3vl' "
            "rewrites it with MiniMax-H3's own conditioner; 'gemma4' with LTX-2.5's bundled prompt enhancer; "
            "'none' uses the raw text (see --prompt_fallback)."
        ),
    )
    group.add_argument(
        "--prompt_fallback",
        choices=["append", "replace"],
        default="append",
        help=(
            "Without the enhancer: 'append' adds the viewer's text to the opening prompt; 'replace' uses the "
            "viewer's text alone (with a continuity line), so the opening prompt's dialogue is not spoken again."
        ),
    )
    group.add_argument(
        "--enhancer_max_new_tokens",
        type=int,
        default=384,
        help="Cap on enhancer output tokens; generation pauses while the enhancer writes.",
    )
    group.add_argument(
        "--transition_chunks",
        type=int,
        default=1,
        help="How many chunks use the smooth TRANSITION prompt before switching to the CONTINUATION prompt.",
    )
    group.add_argument(
        "--cut_in",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Start a new prompt's transition from the frame that will be on screen when it is ready, "
            "dropping the buffered video after it, instead of after the whole buffer has played."
        ),
    )
    group.add_argument(
        "--cut_margin_s",
        type=float,
        default=1.0,
        help="Safety margin added to the estimated chunk time when picking the cut-in frame.",
    )
    group.add_argument(
        "--enhance_estimate_s",
        type=float,
        default=20.0,
        help="Initial prompt-enhancement time estimate; replaced by measured times after the first prompt.",
    )
    return parser


def _broadcast_from_writer(obj):
    """Broadcast one picklable object from the writer (last) rank to all ranks over gloo."""
    world = get_world_group()
    if world.world_size == 1:
        return obj
    box = [obj]
    torch.distributed.broadcast_object_list(box, src=world.ranks[-1], group=world.cpu_group)
    return box[0]


def _shared_condition(frame: np.ndarray) -> np.ndarray:
    """The writer's conditioning frame, broadcast so every rank encodes the same image."""
    world = get_world_group()
    tensor = torch.from_numpy(np.ascontiguousarray(frame)).to(f"cuda:{world.local_rank}")
    world.broadcast(tensor, src=world.world_size - 1)
    return tensor.cpu().numpy()


def _wait_for_playback(stream_dir: Path, generated: int, max_ahead_frames: int, interrupt) -> bool:
    """Block while more than ``max_ahead_frames`` generated frames are unplayed; return whether it blocked.

    Returns early when ``interrupt()`` is true (a viewer prompt arrived), so new
    prompts are picked up at once instead of after the next chunk's turn.
    """
    waited = False
    while generated - read_position(stream_dir) > max_ahead_frames:
        if interrupt():
            break
        waited = True
        time.sleep(0.05)
    return waited


def _publish_state(stream_dir: Path, writer: bool, **state) -> None:
    if writer:
        write_prompt_state(stream_dir, state)


def main() -> None:
    setup_logging()
    args = vars(build_parser().parse_args())
    stream_dir = args["stream_dir"]
    runner = xFuserModelRunner(args)
    runner.print_args(args)

    repo_id = runner.model.settings.model_name
    local_dir = resolve_local_snapshot(repo_id)
    if local_dir is not None:
        localize_modular_components(repo_id, local_dir)
        runner.model.settings.model_name = local_dir
        log(f"Loading {repo_id} from the local HF cache: {local_dir}")
    else:
        log(f"{repo_id} is not in the local HF cache; loading it from the hub.")

    input_args = runner.preprocess_args(args)
    if len(input_args.get("input_images") or []) != 1:
        raise ValueError("Streaming needs exactly one --input_images anchor for the first chunk.")
    width, height = input_args["width"], input_args["height"]
    # Every chunk's conditioning frame comes out at the canvas size. Give the
    # anchor the same size so chunk 0 has the same shapes as the rest: one
    # compiled graph, no recompiles mid-stream. Cover-and-crop so a canvas with
    # a different aspect ratio never stretches the picture.
    source_image = upright(input_args["input_images"][0])
    anchor_image = cover_resize(source_image.convert("RGB"), (width, height))
    image_w, image_h = source_image.size
    if abs(image_w / image_h - width / height) > 0.05:
        log(
            f"The anchor is {image_w}x{image_h} but the canvas is {width}x{height}; it is center-cropped. "
            "Pick a canvas with the anchor's aspect ratio (xfuser/streaming/resolution.py) to keep all of it."
        )
    input_args = with_condition(input_args, anchor_image)
    anchor = np.asarray(anchor_image)

    # Viewer prompts change the text length; compile one graph for all lengths.
    runner.model.dynamic_text_length = True
    runner.initialize(input_args)
    model = runner.model
    model._validate_args(input_args)
    if model.config.warmup_calls:
        model._run_warmup_calls(input_args)

    writer = is_last_process()

    # Prompt enhancer (reuses the resident Qwen3-VL conditioner on the writer rank).
    enhancer = None
    if args["prompt_enhancer"] == "qwen3vl" and model.config.task != "i2va":
        log(f"Prompt enhancement disabled: it writes MiniMax-H3 I2VA prompts, not {model.config.task} prompts.")
    elif args["prompt_enhancer"] == "gemma4":
        try:
            enhancer = LtxPromptEnhancer(model.pipe, max_new_tokens=args["enhancer_max_new_tokens"])
            log("Prompt enhancement enabled (LTX-2.5 Gemma 4).")
        except ValueError as error:
            log(f"Prompt enhancement disabled: {error}")
    elif args["prompt_enhancer"] == "qwen3vl":
        try:
            enhancer = PromptEnhancer(
                model.pipe,
                model_name=model.settings.model_name,
                text_encoder_tp_degree=getattr(model.config, "text_encoder_tp_degree", 1) or 1,
                max_new_tokens=args["enhancer_max_new_tokens"],
            )
            log("Prompt enhancement enabled (Qwen3-VL).")
        except ValueError as error:
            log(f"Prompt enhancement disabled: {error}")
            enhancer = None

    fps = model.settings.fps
    num_frames = input_args["num_frames"]
    budget_s = (num_frames - 1) / fps
    num_chunks = max(1, math.ceil((args["duration_s"] * fps - num_frames) / (num_frames - 1)) + 1)
    transition_chunks = max(0, args["transition_chunks"])
    log(
        f"Streaming {num_chunks} chunks of {num_frames} frames ({budget_s:.3f}s of new video each) "
        f"to {stream_dir}; real-time needs each chunk in under {budget_s:.3f}s."
    )

    original_prompt = input_args["prompt"]
    # Some runners (LTX-2.5) keep --prompt as a list; prompt edits work on one string.
    if isinstance(original_prompt, list):
        if len(original_prompt) != 1:
            raise ValueError(f"Streaming needs exactly one --prompt, got {len(original_prompt)}.")
        original_prompt = original_prompt[0]
    current_prompt = original_prompt
    pending_continuation = None
    transition_remaining = 0
    phase = PHASE_ORIGINAL
    active_prompt_id = 0
    active_user_text = ""
    last_enhance_time = 0.0

    _publish_state(
        stream_dir,
        writer,
        id=0,
        user_text="",
        status=STATE_LIVE,
        phase=PHASE_ORIGINAL,
        first_chunk_index=0,
        enhance_time_s=0.0,
    )

    condition = anchor_image
    loop_times = []
    paced = False
    history = FrameHistory()
    max_ahead_frames = args["max_ahead_chunks"] * (num_frames - 1)
    enhance_estimate = args["enhance_estimate_s"]
    cut_epoch = 0
    for index in range(num_chunks):
        if writer:
            prompt_waiting = lambda: bool(read_new_prompts(stream_dir, active_prompt_id))  # noqa: E731
            if _wait_for_playback(stream_dir, history.end, max_ahead_frames, prompt_waiting) and not paced:
                paced = True
                log(f"Generation is {args['max_ahead_chunks']} chunks ahead of playback; now paced by the streamer.")

        # Pick up a new viewer prompt (writer reads; the choice is broadcast to all ranks).
        chosen = None
        cut = None
        if writer:
            pending = read_new_prompts(stream_dir, active_prompt_id)
            if pending:
                chosen = {"id": pending[-1].id, "text": pending[-1].text}
                if len(pending) > 1:
                    skipped = ", ".join(f"#{item.id}" for item in pending[:-1])
                    log(f"Coalescing queued prompts; using #{chosen['id']} and skipping {skipped}.")
        chosen = _broadcast_from_writer(chosen)

        if chosen is not None:
            active_prompt_id = chosen["id"]
            active_user_text = chosen["text"]
            _publish_state(
                stream_dir,
                writer,
                id=active_prompt_id,
                user_text=active_user_text,
                status=STATE_ENHANCING,
                phase=PHASE_TRANSITION,
                first_chunk_index=index,
                enhance_time_s=0.0,
            )
            # The transition must be ready before playback reaches its first frame:
            # enhancement, then one chunk (slowest recent loop) plus a margin.
            lead_s = (max(loop_times[-5:]) if loop_times else budget_s) + args["cut_margin_s"]
            enhance_start = time.perf_counter()
            if enhancer is not None:
                if writer:
                    # Show the enhancer the frame that should be on screen when the
                    # transition starts; the exact cut frame is fixed after enhancing.
                    reference = np.asarray(condition)
                    if args["cut_in"] and history.end:
                        target = read_position(stream_dir) + math.ceil((enhance_estimate + lead_s) * fps)
                        if target < history.end - 1:
                            reference = history.frame(target)
                    result = enhancer.enhance(original_prompt, current_prompt, active_user_text, reference)
                    enh_payload = {
                        "transition": result.transition,
                        "continuation": result.continuation,
                        "ok": result.ok,
                    }
                else:
                    enh_payload = None
                enh_payload = _broadcast_from_writer(enh_payload)
            else:
                # Build on the original prompt, not the current one, so the text does not
                # grow by one request per prompt over a long stream.
                fallback = PromptEnhancer.fallback(
                    original_prompt, active_user_text, replace=args["prompt_fallback"] == "replace"
                )
                enh_payload = {
                    "transition": fallback.transition,
                    "continuation": fallback.continuation,
                    "ok": False,
                }
            last_enhance_time = time.perf_counter() - enhance_start
            if enhancer is not None:
                enhance_estimate = last_enhance_time
                log(
                    f"Enhanced prompt #{active_prompt_id}:\n--- TRANSITION ---\n{enh_payload['transition']}\n"
                    f"--- CONTINUATION ---\n{enh_payload['continuation']}"
                )

            if writer and args["cut_in"] and history.end:
                played = read_position(stream_dir)
                keep_through = played + math.ceil(lead_s * fps)
                if keep_through < history.end - 1:
                    cut = {"played": played, "keep_through": keep_through, "dropped": history.end - 1 - keep_through}
            cut = _broadcast_from_writer(cut)
            if cut is not None:
                raw = _shared_condition(history.frame(cut["keep_through"]) if writer else np.zeros_like(anchor))
                if args["cond_color_match"] != "none":
                    raw = match_color_stats(
                        raw,
                        anchor,
                        strength=args["cond_color_strength"],
                        match_std=args["cond_color_match"] == "anchor",
                    )
                condition = Image.fromarray(raw)
                if writer:
                    cut_epoch += 1
                    write_cut(stream_dir, cut_epoch, cut["keep_through"], index - 1, active_prompt_id)
                    history.truncate(cut["keep_through"])
                log(
                    f"Cut-in for prompt #{active_prompt_id}: playback is at frame {cut['played']}; the transition "
                    f"continues from frame {cut['keep_through']} ({lead_s:.1f}s ahead), dropping "
                    f"{cut['dropped'] / fps:.1f}s of buffered video."
                )
            elif args["cut_in"] and index > 0:
                log("The buffer is shorter than the transition lead time; continuing from the newest frame.")

            if transition_chunks > 0:
                current_prompt = enh_payload["transition"]
                pending_continuation = enh_payload["continuation"]
                transition_remaining = transition_chunks
                phase = PHASE_TRANSITION
            else:
                current_prompt = enh_payload["continuation"]
                pending_continuation = None
                transition_remaining = 0
                phase = PHASE_CONTINUATION
            _publish_state(
                stream_dir,
                writer,
                id=active_prompt_id,
                user_text=active_user_text,
                status=STATE_RENDERING,
                phase=phase,
                first_chunk_index=index,
                enhance_time_s=round(last_enhance_time, 3),
            )
            log(f"Applying prompt #{active_prompt_id} ({phase}) after {last_enhance_time:.2f}s: {active_user_text!r}")

        get_world_group().barrier()

        loop_start = time.perf_counter()
        chunk_args = dict(
            with_condition(input_args, condition),
            prompt=current_prompt,
            seed=input_args["seed"] + index,
        )
        output, gen_time = model._run_timed_pipe(chunk_args)
        frames, audio, sample_rate = chunk_media(model, output)

        last_raw = _shared_condition(frames[-1])
        # The last transition chunk defines the new scene's look. Re-base the color
        # anchor on it before matching, or the first continuation chunk would be
        # conditioned on a frame pulled back toward the old scene's colors.
        if args["cond_color_match"] != "none" and transition_remaining == 1 and pending_continuation is not None:
            anchor = last_raw
        last = last_raw
        if args["cond_color_match"] != "none":
            last = match_color_stats(
                last_raw,
                anchor,
                strength=args["cond_color_strength"],
                match_std=args["cond_color_match"] == "anchor",
            )
        condition = Image.fromarray(last)

        if writer:
            write_chunk(
                stream_dir,
                Chunk(
                    index=index,
                    frames=frames,
                    audio=audio,
                    sample_rate=sample_rate,
                    fps=fps,
                    meta={
                        "gen_time_s": round(gen_time, 4),
                        "loop_time_s": round(time.perf_counter() - loop_start, 4),
                        "budget_s": round(budget_s, 4),
                        "seed": chunk_args["seed"],
                        "steps": input_args["num_inference_steps"],
                        "prompt_id": active_prompt_id,
                        "user_text": active_user_text,
                        "prompt_phase": phase,
                        "enhance_time_s": round(last_enhance_time, 4) if chosen is not None else 0.0,
                        "start_frame": history.end,
                        "cut_in_frame": cut["keep_through"] if cut else None,
                    },
                ),
            )
            history.append(frames[1:] if index > 0 else frames)
            history.prune(read_position(stream_dir))

        if transition_remaining > 0:
            transition_remaining -= 1
            if transition_remaining == 0 and pending_continuation is not None:
                current_prompt = pending_continuation
                pending_continuation = None
                phase = PHASE_CONTINUATION

        loop_times.append(time.perf_counter() - loop_start)
        log(
            f"Chunk {index + 1}/{num_chunks}: pipe {gen_time:.2f}s, loop {loop_times[-1]:.2f}s "
            f"(budget {budget_s:.2f}s, {budget_s / loop_times[-1]:.2f}x real time)"
        )

    if writer:
        mark_done(stream_dir, num_chunks)
    steady = sorted(loop_times)
    log(
        f"Chunk loop time p50 {steady[len(steady) // 2]:.2f}s, "
        f"p95 {steady[min(len(steady) - 1, int(0.95 * len(steady)))]:.2f}s, max {steady[-1]:.2f}s "
        f"against a {budget_s:.2f}s budget."
    )
    runner.cleanup()


if __name__ == "__main__":
    main()
