"""CPU stand-in for the MiniMax-H3 generator, for trying the stream and web UI without GPUs.

Writes chunks to the stream directory in the generator's format and at its
pace (each chunk takes ``--gen_s`` and generation stays ``--max_ahead_chunks``
ahead of playback), and picks up prompts from the same queue. Every frame shows
a moving test pattern over the anchor image with the chunk index and the active
prompt burned in; the soundtrack is a tone whose pitch follows the prompt, so a
bad seam is visible and audible. Prompts apply from the next chunk; there is no
enhancer or cut-in.

    python -m xfuser.streaming.ts_server --stream_dir /dev/shm/h3_stream --width 896 --height 576 &
    python -m xfuser.streaming.fake_generator --stream_dir /dev/shm/h3_stream --width 896 --height 576 \\
        --anchor assets/amd_gpu_keynote.jpg
"""

from __future__ import annotations

import argparse
import textwrap
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from xfuser.streaming.chunks import Chunk, mark_done, read_position, write_chunk
from xfuser.streaming.prompt_queue import (
    STATE_ENHANCING,
    STATE_RENDERING,
    read_new_prompts,
    write_prompt_state,
)
from xfuser.streaming.resolution import cover_resize, upright


class FakeGenerator:
    def __init__(
        self, width: int, height: int, num_frames: int, fps: int, sample_rate: int, anchor: str | None
    ) -> None:
        self.width, self.height = width, height
        self.num_frames, self.fps, self.sample_rate = num_frames, fps, sample_rate
        if anchor:
            self.background = np.asarray(cover_resize(upright(Image.open(anchor)).convert("RGB"), (width, height)))
        else:
            x = np.linspace(0, 1, width)[None, :]
            y = np.linspace(0, 1, height)[:, None]
            self.background = np.stack(np.broadcast_arrays(40 + 120 * x, 30 + 80 * y, 90 + 0 * x * y), -1).astype(
                np.uint8
            )
        self.font = ImageFont.load_default(size=max(18, height // 24))
        self.frame = 0  # global frame counter; each chunk's frame 0 repeats the previous chunk's last frame

    def chunk(self, index: int, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        tint = np.array([(zlib.crc32(prompt.encode()) >> shift) & 0xFF for shift in (0, 8, 16)], dtype=np.float32)
        base = (0.7 * self.background + 0.3 * tint).astype(np.uint8)
        text = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text)
        lines = [f"fake generator - chunk {index}"] + textwrap.wrap(prompt, width=60)[:4]
        for row, line in enumerate(lines):
            y = self.height - 16 - (len(lines) - row) * (self.font.size + 6)
            draw.text((16, y), line, font=self.font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        text = np.asarray(text)
        alpha = text[..., 3:4].astype(np.float32) / 255

        frames = np.empty((self.num_frames, self.height, self.width, 3), dtype=np.uint8)
        radius = self.height // 12
        span = self.width - 2 * radius
        yy, xx = np.ogrid[: self.height, : self.width]
        for i in range(self.num_frames):
            t = self.frame + i
            frame = base.copy()
            cx = radius + abs((t * 9) % (2 * span) - span)
            cy = self.height // 3
            frame[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2] = (250, 220, 60)
            frames[i] = (frame * (1 - alpha) + text[..., :3] * alpha).astype(np.uint8)
        start = self.frame
        self.frame += self.num_frames - 1

        samples = round(self.num_frames * self.sample_rate / self.fps) + 267
        seconds = start / self.fps + np.arange(samples) / self.sample_rate
        pitch = 220.0 * 2 ** ((zlib.crc32(prompt.encode()) % 24) / 12)
        tone = 0.15 * (0.6 + 0.4 * np.cos(2 * np.pi * seconds)) * np.sin(2 * np.pi * pitch * seconds)
        return frames, np.stack([tone, tone]).astype(np.float32)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream_dir", type=Path, default=Path("/dev/shm/h3_stream"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--num_frames", type=int, default=124)
    parser.add_argument("--anchor", default=None, help="Background image (default: a gradient).")
    parser.add_argument("--prompt", default="Fake opening scene")
    parser.add_argument("--gen_s", type=float, default=2.8, help="Seconds each chunk takes, like the real model.")
    parser.add_argument("--enhance_s", type=float, default=2.0, help="Pause for each new prompt, like the enhancer.")
    parser.add_argument("--max_ahead_chunks", type=int, default=4)
    parser.add_argument("--num_chunks", type=int, default=0, help="Stop after this many chunks (0: run forever).")
    args = parser.parse_args(argv)

    generator = FakeGenerator(args.width, args.height, args.num_frames, args.fps, args.sample_rate, args.anchor)
    max_ahead = args.max_ahead_chunks * (args.num_frames - 1)
    prompt, prompt_id, generated, index = args.prompt, 0, 0, 0
    while not args.num_chunks or index < args.num_chunks:
        while generated - read_position(args.stream_dir) > max_ahead:
            time.sleep(0.05)
        meta = {"gen_time_s": args.gen_s, "fake": True}
        pending = read_new_prompts(args.stream_dir, prompt_id)
        if pending:
            prompt_id, prompt = pending[-1].id, pending[-1].text
            state = {"id": prompt_id, "user_text": prompt, "first_chunk_index": index}
            write_prompt_state(args.stream_dir, {**state, "status": STATE_ENHANCING})
            time.sleep(args.enhance_s)
            write_prompt_state(args.stream_dir, {**state, "status": STATE_RENDERING})
            meta["prompt_id"] = prompt_id
        started = time.monotonic()
        frames, audio = generator.chunk(index, prompt)
        time.sleep(max(0.0, args.gen_s - (time.monotonic() - started)))
        write_chunk(args.stream_dir, Chunk(index, frames, audio, args.sample_rate, args.fps, meta))
        generated += len(frames) if index == 0 else len(frames) - 1
        index += 1
    mark_done(args.stream_dir, index)


if __name__ == "__main__":
    main()
