"""Interactive prompt console for a running MiniMax-H3 stream.

Type a new prompt and press Enter to queue it; the generator rewrites it with
the model's own Qwen3-VL conditioner (using the frame the transition starts
from for continuity) and cuts the transition in a few seconds ahead of playback.
The prompt directory is shared with the generator and the streamer through
``--stream_dir``.

    python -m xfuser.streaming.prompt_cli --stream_dir /dev/shm/h3_stream [--demo]

With ``--demo``, a scripted prompt (see ``xfuser.streaming.demo``) is queued
whenever no prompt has arrived for ``--demo_interval_s``; typed prompts still
work and restart that countdown. Prompts from elsewhere (the web page, the demo,
another console) are printed as they arrive. ``--no_console`` drops the input
and only prints them (and runs the demo with ``--demo``).

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
from collections.abc import Callable
from pathlib import Path

from xfuser.streaming.demo import DemoDirector, run_demo
from xfuser.streaming.prompt_queue import (
    UserPrompt,
    append_prompt,
    latest_prompt,
    read_new_prompts,
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


def _start_demo(args, lock: threading.Lock, stop: threading.Event) -> threading.Thread:
    cast = [name.strip() for name in args.demo_cast.split(",") if name.strip()] if args.demo_cast else []
    director = DemoDirector(cast=cast, seed=args.demo_seed)
    thread = threading.Thread(
        target=run_demo,
        kwargs=dict(
            stream_dir=args.stream_dir,
            director=director,
            stop=stop,
            interval_s=args.demo_interval_s,
            lock=lock,
        ),
        name="demo",
        daemon=True,
    )
    thread.start()
    return thread


def watch_prompts(
    stream_dir: Path,
    stop: threading.Event,
    announce: Callable[[UserPrompt], None],
    skip: set[int] | None = None,
    lock: threading.Lock | None = None,
    poll_s: float = 0.5,
) -> None:
    """Call ``announce`` for each prompt queued after this starts, except ids in ``skip``.

    Hold ``lock`` while queueing a prompt and adding its id to ``skip`` so it is never announced.
    """
    lock = lock or threading.Lock()
    latest = latest_prompt(stream_dir)
    seen = latest.id if latest else 0
    while not stop.wait(poll_s):
        with lock:
            new = read_new_prompts(stream_dir, seen)
            if new:
                seen = new[-1].id
            new = [prompt for prompt in new if not skip or prompt.id not in skip]
        for prompt in new:
            announce(prompt)


def _start_feed(
    stream_dir: Path, stop: threading.Event, skip: set[int], interactive: bool, lock: threading.Lock | None = None
) -> threading.Thread:
    def announce(prompt: UserPrompt) -> None:
        print(f"\n[{prompt.source}] queued prompt #{prompt.id}: {prompt.text}", flush=True)
        if interactive and "readline" in sys.modules:
            sys.modules["readline"].redisplay()

    thread = threading.Thread(
        target=watch_prompts, args=(stream_dir, stop, announce, skip, lock), name="feed", daemon=True
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
        help="No input: print prompts as they are queued (and run the demo with --demo) until interrupted.",
    )
    parser.add_argument("--quiet", action="store_true", help="With --no_console, do not print queued prompts.")
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
        if args.quiet and not args.demo:
            parser.error("--no_console --quiet does nothing without --demo")
        if args.demo:
            print(f"Demo mode for {args.stream_dir}: a new prompt after {args.demo_interval_s:.0f}s without one.")
            _start_demo(args, lock, stop)
        if not args.quiet:
            _start_feed(args.stream_dir, stop, skip=set(), interactive=False)
        try:
            stop.wait()
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
        _start_demo(args, lock, stop)
    typed: set[int] = set()
    _start_feed(args.stream_dir, stop, skip=typed, interactive=True, lock=lock)
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
            typed.add(prompt.id)
        print(f"  queued prompt #{prompt.id}; it cuts in once it is enhanced and rendered.")
    stop.set()
    print("Console closed; the stream keeps running.", file=sys.stderr)


if __name__ == "__main__":
    main()
