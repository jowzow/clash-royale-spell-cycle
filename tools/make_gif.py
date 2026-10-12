#!/usr/bin/env python3
"""
Turns a slice of an annotated clip into the GIF the README shows inline.

GitHub will not play an mp4 embedded in markdown, so the demo at the top of
the README has to be a GIF. This keeps that asset reproducible instead of
being something that was made once by hand and can never be regenerated.

Usage:
    python tools/make_gif.py output/videoplayback_spellcycle.mp4 --start 12 --seconds 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: Pillow is needed for GIF output. pip install pillow")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out", type=Path, default=Path("output/demo.gif"))
    p.add_argument("--start", type=float, default=0.0,
                   help="seconds into the clip to begin")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--scale", type=float, default=0.70)
    p.add_argument("--colors", type=int, default=96)
    args = p.parse_args()

    path = Path(args.video)
    if not path.exists():
        sys.exit(f"ERROR: no such video: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"ERROR: OpenCV could not open {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / args.fps))
    first = int(args.start * src_fps)
    last = first + int(args.seconds * src_fps)

    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok or i >= last:
            break
        if i >= first and (i - first) % step == 0:
            fr = cv2.resize(fr, None, fx=args.scale, fy=args.scale,
                            interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(
                cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).quantize(colors=args.colors))
        i += 1
    cap.release()

    if not frames:
        sys.exit("ERROR: no frames selected. Check --start against the clip "
                 "length.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    mb = args.out.stat().st_size / 1_048_576
    print(f"  {args.out}: {len(frames)} frames, {frames[0].size[0]}x"
          f"{frames[0].size[1]}, {mb:.1f} MB")
    if mb > 10:
        print("  warning: over 10 MB, GitHub will be slow to render it. "
              "Lower --seconds, --scale or --colors.")


if __name__ == "__main__":
    main()
