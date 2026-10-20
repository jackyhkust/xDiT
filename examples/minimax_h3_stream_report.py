"""Drift report for one or more MiniMax-H3 stream runs.

    python examples/minimax_h3_stream_report.py /output/stream_5min /output/stream_5min_cm --out /output/report

Each run directory needs the streamer's ``metrics.jsonl`` and ``stream.ts``.
Writes ``drift_metrics.png`` (runs overlaid, one measure per panel) and a
``contact_sheet_<run>.png`` per run (one frame every 30 s), and prints first-
versus last-minute averages plus playout checks of the recording.
"""

import argparse
import json
from pathlib import Path

import av
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FPS = 24
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
PANELS = [
    ("L* mean − anchor (lightness)", lambda r: r["lab_mean_delta"][0], 0.0),
    ("b* mean − anchor (warm/cool cast)", lambda r: r["lab_mean_delta"][2], 0.0),
    ("a* std / anchor (red–green spread)", lambda r: r["lab_std_ratio"][1], 1.0),
    ("b* std / anchor (blue–yellow spread)", lambda r: r["lab_std_ratio"][2], 1.0),
    ("Sharpness / anchor (Laplacian variance)", lambda r: r["sharpness_ratio"], 1.0),
    ("Seam PSNR, dB (conditioning frame vs new frame 0)", lambda r: r.get("seam_psnr", np.nan), None),
]


def content_frames(num_chunks: int, num_frames: int) -> int:
    # Chunk 0 is shown whole; later chunks add num_frames - 1.
    return num_frames + (num_chunks - 1) * (num_frames - 1)


def load_metrics(run: Path, num_frames: int):
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    minutes = np.array([content_frames(row["index"] + 1, num_frames) / FPS / 60 for row in rows])
    return rows, minutes


def plot(runs, out: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.5), facecolor=SURFACE)
    for ax, (title, get, reference) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        if reference is not None:
            ax.axhline(reference, color=INK_SECONDARY, lw=1, ls="--")
        for color, (name, rows, minutes) in zip(SERIES, runs):
            values = np.array([get(row) for row in rows], dtype=float)
            ax.plot(minutes, values, color=color, lw=2, marker="o", ms=3, label=name)
            last = np.flatnonzero(~np.isnan(values))[-1]
            ax.annotate(
                name,
                (minutes[last], values[last]),
                xytext=(4, 0),
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=8,
                va="center",
            )
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("stream time (min), last frame of each chunk", color=INK_SECONDARY, fontsize=9)
        ax.margins(x=0.12)
    if len(runs) > 1:
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper right", frameon=False, labelcolor=INK)
    fig.suptitle(
        "Per-chunk drift vs. the anchor image (dashed = anchor value)", color=INK, fontsize=12, x=0.01, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "drift_metrics.png", dpi=110, facecolor=SURFACE)


def check_recording(name: str, run: Path, num_content_frames: int, out: Path) -> None:
    container = av.open(str(run / "stream.ts"))
    stream = container.streams.video[0]
    frames, pts = [], []
    for frame in container.decode(video=0):
        pts.append(float(frame.pts * stream.time_base))
        frames.append(frame)
    gaps = np.diff(pts)
    container = av.open(str(run / "stream.ts"))
    audio_s = sum(frame.samples for frame in container.decode(audio=0)) / 32000
    print(
        f"{name}: {len(pts)} frames, {1 / gaps.mean():.4f} fps, frame gap {gaps.min():.4f}-{gaps.max():.4f}s, "
        f"video {len(pts) / FPS:.1f}s / audio {audio_s:.1f}s"
    )

    start = len(frames) - num_content_frames
    tiles = []
    for second in range(0, num_content_frames // FPS + 1, 30):
        tile = frames[start + second * FPS].to_image().resize((320, 181))
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, 40, 16], fill=(0, 0, 0))
        draw.text((4, 2), f"{second // 60}:{second % 60:02d}", fill=(255, 255, 255))
        tiles.append(tile)
    columns = 4
    sheet = Image.new("RGB", (320 * columns, 181 * -(-len(tiles) // columns)))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * 320, (i // columns) * 181))
    sheet.save(out / f"contact_sheet_{name}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num_frames", type=int, default=124, help="Frames per generated chunk.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    for run in args.runs:
        rows, minutes = load_metrics(run, args.num_frames)
        runs.append((run.name, rows, minutes))
        print(
            f"\n{run.name}: {len(rows)} chunks; gen time p50 "
            f"{np.median([r['gen_time_s'] for r in rows]):.2f}s, max {max(r['gen_time_s'] for r in rows):.2f}s"
        )
        first, last = minutes <= 1.0, minutes >= minutes[-1] - 1.0
        for title, get, _ in PANELS:
            values = np.array([get(row) for row in rows], dtype=float)
            print(
                f"  {title:52s} first min {np.nanmean(values[first]):8.3f}   last min {np.nanmean(values[last]):8.3f}"
            )
        check_recording(run.name, run, content_frames(len(rows), args.num_frames), args.out)
    plot(runs, args.out)


if __name__ == "__main__":
    main()
