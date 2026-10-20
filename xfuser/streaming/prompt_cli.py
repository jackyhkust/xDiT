"""Interactive prompt console for a running MiniMax-H3 stream.

Type a new prompt and press Enter to queue it; the generator rewrites it with
the model's own Qwen3-VL conditioner (using the frame the transition starts
from for continuity) and cuts the transition in a few seconds ahead of playback.
The prompt directory is shared with the generator and the streamer through
``--stream_dir``.

    python -m xfuser.streaming.prompt_cli --stream_dir /dev/shm/h3_stream [--demo]

With ``--demo``, a scripted prompt (see ``xfuser.streaming.demo``) is queued
whenever no prompt has arrived for ``--demo_interval_s``; typed prompts still
work and restart that countdown.

Commands:
    <text>        queue <text> as the next prompt
    /overlay      switch the video overlay (buffered frames, latest input) on or off
    /status       print the generator's prompt state
    /quit, /exit  leave the console (the stream keeps running)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from pathlib import Path

from xfuser.streaming.demo import DemoDirector, run_demo
from xfuser.streaming.prompt_queue import (
    UserPrompt,
    append_prompt,
    read_prompt_state,
    toggle_overlay,
)

# Terminal escape sequences (arrow keys etc.) that leak in when line editing is unavailable.
_ESCAPES = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-_]|[\x00-\x08\x0b-\x1f\x7f]")


def clean_prompt(text: str) -> str:
    return " ".join(_ESCAPES.sub("", text).split())


def _print_status(stream_dir: Path) -> None:
    state = read_prompt_state(stream_dir)
    if not state:
        print("(no prompt state yet; the generator may still be loading)")
        return
    print(json.dumps(state, indent=2))


def _start_demo(args, lock: threading.Lock, stop: threading.Event, interactive: bool) -> threading.Thread:
    cast = [name.strip() for name in args.demo_cast.split(",") if name.strip()] if args.demo_cast else []
    director = DemoDirector(cast=cast, seed=args.demo_seed)

    def announce(prompt: UserPrompt) -> None:
        print(f"\n[demo] queued prompt #{prompt.id}: {prompt.text}", flush=True)
        if interactive and "readline" in sys.modules:
            sys.modules["readline"].redisplay()

    thread = threading.Thread(
        target=run_demo,
        kwargs=dict(
            stream_dir=args.stream_dir,
            director=director,
            stop=stop,
            interval_s=args.demo_interval_s,
            on_prompt=announce,
            lock=lock,
        ),
        name="demo",
        daemon=True,
    )
    thread.start()
    return thread


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Queue prompts for a running MiniMax-H3 stream.")
    parser.add_argument("--stream_dir", type=Path, default=Path("/dev/shm/h3_stream"))
    parser.add_argument("--prompt", default=None, help="Queue one prompt and exit instead of starting the console.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Queue a scripted AMD-promo prompt whenever no prompt has arrived for --demo_interval_s.",
    )
    parser.add_argument("--demo_interval_s", type=float, default=30.0)
    parser.add_argument(
        "--demo_cast",
        default="",
        help="Comma-separated characters already in the opening scene, counted toward the three-character cast.",
    )
    parser.add_argument("--demo_seed", type=int, default=None, help="Seed for repeatable demo prompts.")
    parser.add_argument(
        "--no_console",
        action="store_true",
        help="With --demo, run only the demo loop (no input) until interrupted.",
    )
    args = parser.parse_args(argv)

    if args.prompt is not None:
        if clean_prompt(args.prompt) == "/overlay":
            print(f"overlay {'on' if toggle_overlay(args.stream_dir) else 'off'}")
            return
        prompt = append_prompt(args.stream_dir, clean_prompt(args.prompt))
        print(f"queued prompt #{prompt.id}: {prompt.text}")
        return

    lock = threading.Lock()
    stop = threading.Event()
    if args.no_console:
        if not args.demo:
            parser.error("--no_console needs --demo")
        print(f"Demo mode for {args.stream_dir}: a new prompt every {args.demo_interval_s:.0f}s of quiet.")
        demo = _start_demo(args, lock, stop, interactive=False)
        try:
            demo.join()
        except KeyboardInterrupt:
            stop.set()
        return

    try:
        import readline  # noqa: F401  (arrow-key editing and history for input())
    except ImportError:
        pass
    print(f"Interactive prompt console for {args.stream_dir}.")
    print("Type a prompt and press Enter. /overlay toggles the video overlay, /status shows state, /quit leaves.")
    if args.demo:
        print(
            f"Demo mode: a scripted prompt is queued after {args.demo_interval_s:.0f}s without a new prompt "
            "(the countdown starts once the video plays)."
        )
        _start_demo(args, lock, stop, interactive=True)
    print()
    while True:
        try:
            line = clean_prompt(input("prompt> "))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/status":
            _print_status(args.stream_dir)
            continue
        if line == "/overlay":
            enabled = toggle_overlay(args.stream_dir)
            print(f"  overlay {'on' if enabled else 'off'}.")
            continue
        with lock:
            prompt = append_prompt(args.stream_dir, line)
        print(f"  queued prompt #{prompt.id}; it cuts in once it is enhanced and rendered.")
    stop.set()
    print("Console closed; the stream keeps running.", file=sys.stderr)


if __name__ == "__main__":
    main()
