"""Continuous MiniMax-H3 I2V generation for a live stream.

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

To stay real-time, a chunk must take no longer to generate than its new frames
take to play: ``(num_frames - 1) / fps`` seconds.
"""

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
from xfuser.streaming.chunks import Chunk, mark_done, read_played, write_chunk
from xfuser.streaming.color import match_color_stats


def build_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(description="Continuous MiniMax-H3 I2V generation for streaming")
    xFuserArgs.add_runner_args(parser)
    group = parser.add_argument_group("Streaming")
    group.add_argument("--stream_dir", type=Path, default=Path("/dev/shm/h3_stream"))
    group.add_argument("--duration_s", type=float, default=330.0, help="Stream length to generate, in seconds.")
    group.add_argument(
        "--max_ahead_chunks",
        type=int,
        default=3,
        help="Pause generation while this many written chunks are still waiting to play.",
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
    return parser


def _wait_for_playback(stream_dir: Path, written: int, max_ahead: int) -> bool:
    """Block while more than ``max_ahead`` written chunks are unplayed; return whether it blocked."""
    waited = False
    while written - read_played(stream_dir) > max_ahead:
        waited = True
        time.sleep(0.05)
    return waited


def _shared_condition(frame: np.ndarray) -> np.ndarray:
    """The writer's conditioning frame, broadcast so every rank encodes the same image."""
    world = get_world_group()
    tensor = torch.from_numpy(np.ascontiguousarray(frame)).to(f"cuda:{world.local_rank}")
    world.broadcast(tensor, src=world.world_size - 1)
    return tensor.cpu().numpy()


def main() -> None:
    setup_logging()
    args = vars(build_parser().parse_args())
    stream_dir = args["stream_dir"]
    runner = xFuserModelRunner(args)
    runner.print_args(args)

    input_args = runner.preprocess_args(args)
    if len(input_args.get("input_images") or []) != 1:
        raise ValueError("Streaming needs exactly one --input_images anchor for the first chunk.")
    width, height = input_args["width"], input_args["height"]
    # Every chunk's conditioning frame comes out at the canvas size. Give the
    # anchor the same size so chunk 0 has the same shapes as the rest: one
    # compiled graph, no recompiles mid-stream.
    anchor_image = input_args["input_images"][0].convert("RGB")
    if anchor_image.size != (width, height):
        anchor_image = anchor_image.resize((width, height), Image.Resampling.LANCZOS)
    input_args["input_images"] = [anchor_image]
    anchor = np.asarray(anchor_image)

    runner.initialize(input_args)
    model = runner.model
    model._validate_args(input_args)
    if model.config.warmup_calls:
        model._run_warmup_calls(input_args)

    fps = model.settings.fps
    num_frames = input_args["num_frames"]
    budget_s = (num_frames - 1) / fps
    num_chunks = max(1, math.ceil((args["duration_s"] * fps - num_frames) / (num_frames - 1)) + 1)
    writer = is_last_process()
    log(
        f"Streaming {num_chunks} chunks of {num_frames} frames ({budget_s:.3f}s of new video each) "
        f"to {stream_dir}; real-time needs each chunk in under {budget_s:.3f}s."
    )

    condition = anchor_image
    loop_times = []
    paced = False
    for index in range(num_chunks):
        if writer and _wait_for_playback(stream_dir, index, args["max_ahead_chunks"]) and not paced:
            paced = True
            log(f"Generation is {args['max_ahead_chunks']} chunks ahead of playback; now paced by the streamer.")
        get_world_group().barrier()

        loop_start = time.perf_counter()
        chunk_args = dict(input_args, input_images=[condition], seed=input_args["seed"] + index)
        output, gen_time = model._run_timed_pipe(chunk_args)
        frames = output.videos[0].numpy()
        audio = output.audio[0].float().cpu().numpy() if output.audio is not None else None

        last = _shared_condition(frames[-1])
        if args["cond_color_match"] != "none":
            last = match_color_stats(
                last,
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
                    sample_rate=output.audio_sample_rate,
                    fps=fps,
                    meta={
                        "gen_time_s": round(gen_time, 4),
                        "loop_time_s": round(time.perf_counter() - loop_start, 4),
                        "budget_s": round(budget_s, 4),
                        "seed": chunk_args["seed"],
                        "steps": input_args["num_inference_steps"],
                    },
                ),
            )
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
