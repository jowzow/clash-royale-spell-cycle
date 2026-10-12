#!/usr/bin/env python3
"""
Dumps frames from a gameplay video so the UI regions can be measured.

The hand and the tower health bars sit at fixed fractions of the screen, but
which fractions depends on the recording (phone, emulator, letterboxed
YouTube rip). Run this once on a new video, look at the frames it writes,
and set the regions in spellcycle.py accordingly.

Usage:
    python tools/inspect_frame.py input/match.mp4
    python tools/inspect_frame.py input/match.mp4 --at 12.5 --grid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def main():
    p = argparse.ArgumentParser(description="Dump frames for calibration.")
    p.add_argument("video")
    p.add_argument("--at", type=float, nargs="*",
                   help="timestamps in seconds (default: 6 spread through)")
    p.add_argument("--grid", action="store_true",
                   help="overlay a 10%% grid with labels")
    p.add_argument("--outdir", default="output/frames")
    args = p.parse_args()

    path = Path(args.video)
    if not path.exists():
        sys.exit(f"ERROR: no such video: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"ERROR: OpenCV could not open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = total / fps if fps else 0
    print(f"  {path.name}: {w}x{h}, {fps:.2f} fps, {total} frames, {dur:.1f}s")

    stamps = args.at if args.at else [dur * f for f in
                                      (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for t in stamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total - 1, int(t * fps))))
        ok, frame = cap.read()
        if not ok:
            print(f"  could not read frame at {t:.1f}s")
            continue

        if args.grid:
            for i in range(1, 10):
                x, y = w * i // 10, h * i // 10
                cv2.line(frame, (x, 0), (x, h), (0, 255, 255), 1)
                cv2.line(frame, (0, y), (w, y), (0, 255, 255), 1)
                cv2.putText(frame, f"{i * 10}", (x + 2, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(frame, f"{i * 10}", (2, y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        out = outdir / f"t{t:06.2f}.jpg"
        cv2.imwrite(str(out), frame)
        print(f"  wrote {out}")

    cap.release()


if __name__ == "__main__":
    main()
