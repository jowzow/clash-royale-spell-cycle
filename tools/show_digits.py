#!/usr/bin/env python3
"""
Draws what the health reader sees, for the README.

One row per sample: the strip of frame as it comes off the video, the ink
mask the reader works on after normalising and dropping anything strongly
coloured, and the number it decoded with its score.

Usage:
    python tools/show_digits.py videoplayback.mp4
    python tools/show_digits.py videoplayback.mp4 --at 600,645,679,680
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from digits import BAND, SPAN, DigitReader, prep   # noqa: E402

ZOOM = 9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--at", default="600,645,660,679",
                   help="comma separated seconds to sample")
    p.add_argument("--layout", type=Path, default=ROOT / "data" / "layout.json")
    p.add_argument("--digits", type=Path, default=ROOT / "data" / "digits.json")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "health_read.png")
    args = p.parse_args()

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"ERROR: no such video: {video}")
    anchors = json.loads(args.layout.read_text(encoding="utf-8"))["enemy_hp_text"]
    reader = DigitReader.load(args.digits)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"ERROR: OpenCV could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    rows = []
    for sec in [float(x) for x in args.at.split(",")]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        for side, x0 in anchors.items():
            strip = frame[BAND[0]:BAND[1], int(x0):int(x0) + SPAN]
            value, score = reader.read(strip)
            _soft, mask = prep(strip)

            shown = cv2.resize(strip, None, fx=ZOOM, fy=ZOOM,
                               interpolation=cv2.INTER_NEAREST)
            if mask is None:
                ink = np.zeros_like(shown)
            else:
                ink = cv2.resize((mask * 255).astype(np.uint8), None,
                                 fx=ZOOM, fy=ZOOM,
                                 interpolation=cv2.INTER_NEAREST)
                ink = cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)
                pad = shown.shape[0] - ink.shape[0]
                ink = cv2.copyMakeBorder(ink, pad // 2, pad - pad // 2, 0, 0,
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))

            label = np.zeros((shown.shape[0], 330, 3), np.uint8)
            cv2.putText(label, f"{sec:.0f}s {side.lower()}", (8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (160, 160, 160), 1,
                        cv2.LINE_AA)
            text = "unreadable" if value is None else str(value)
            colour = (90, 90, 230) if value is None else (90, 230, 120)
            cv2.putText(label, text, (8, 86), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        colour, 2, cv2.LINE_AA)
            cv2.putText(label, f"{score:.2f}", (190, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (150, 150, 150), 1,
                        cv2.LINE_AA)
            gap = np.zeros((shown.shape[0], 14, 3), np.uint8)
            rows.append(cv2.copyMakeBorder(
                np.hstack([label, gap, shown, gap, ink]), 0, 6, 6, 6,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    cap.release()

    if not rows:
        sys.exit("ERROR: no frames sampled. Check --at against the clip length.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows))
    print(f"  wrote {args.out}  ({len(rows)} strips)")


if __name__ == "__main__":
    main()
