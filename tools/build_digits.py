#!/usr/bin/env python3
"""
Cuts the ten digit glyphs out of a recording and writes data/digits.json.

Supercell's font is fixed, so one glyph picture per digit is enough to read
every health number in the match. The pictures come from frames whose value
was read by eye and written down in data/hp_labels.json; the known value
says how many digits are in the strip, which is all the splitter needs to
cut it into glyphs.

The templates are built twice. The first pass gives a rough set, the second
uses them to work out which row each training strip's text really sits on
and re-cuts at that offset. Without it a one-row slip in a handful of frames
blurs the averages enough that an 8 starts matching everything.

Rows marked "test" in the label file never contribute a glyph, so the
accuracy printed at the end is measured on numbers the templates have not
seen.

Usage:
    python tools/build_digits.py videoplayback.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from digits import (BAND, DIGITS, HALF, SPAN, TWU, Z, DigitReader, centroid,
                    extent, ncc, prep, split, upscale)


def strip_at(cap, fps, sec, x0):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame[BAND[0]:BAND[1], x0:x0 + SPAN].copy()


def glyphs(bgr, label, dy=0):
    """-> list of (digit, upscaled patch, advance to the next digit)."""
    soft, mask = prep(bgr, dy)
    if soft is None:
        return []
    found = extent(mask)
    if found is None:
        return []
    lo, hi = found
    cuts = split(mask, lo, hi, len(label))
    if cuts is None:
        return []
    up = upscale(soft)
    centres = [centroid(mask, a, b) for a, b in cuts]
    out = []
    for i, (d, c) in enumerate(zip(label, centres)):
        left = int(round((c - HALF) * Z))
        if left < 0 or left + TWU > up.shape[1]:
            continue
        step = centres[i + 1] - c if i + 1 < len(centres) else None
        out.append((d, up[:, left:left + TWU], step))
    return out


def assemble(samples):
    bank, pitch = defaultdict(list), defaultdict(list)
    for d, patch, step in samples:
        bank[d].append(patch)
        if step is not None:
            pitch[d].append(step)
    templates = {d: np.mean(v, axis=0) for d, v in bank.items()}
    advances = {d: (float(np.mean(pitch[d])) if pitch.get(d) else 5.33)
                for d in DIGITS}
    return templates, advances, {d: len(v) for d, v in bank.items()}


def harvest(cap, fps, rows, anchors, offsets=None):
    samples = []
    for row in rows:
        for side, x0 in anchors.items():
            label = row[side.lower()]
            dy = (offsets or {}).get((row["t"], side), 0)
            samples += glyphs(strip_at(cap, fps, row["t"], x0), label, dy)
    return samples


def realign(reader, cap, fps, rows, anchors):
    """Pick the row offset that makes each training strip agree best with the
    first-pass templates."""
    offsets, moved = {}, 0
    for row in rows:
        for side, x0 in anchors.items():
            label = row[side.lower()]
            bgr = strip_at(cap, fps, row["t"], x0)
            best = (0, -9.0)
            for dy in (-1, 0, 1):
                got = glyphs(bgr, label, dy)
                if len(got) != len(label):
                    continue
                s = float(np.mean([ncc(p, reader.t[d])
                                   for d, p, _ in got if d in reader.t]))
                if s > best[1]:
                    best = (dy, s)
            offsets[(row["t"], side)] = best[0]
            moved += best[0] != 0
    print(f"  realigned {moved} of {len(offsets)} training strips")
    return offsets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--labels", type=Path, default=ROOT / "data" / "hp_labels.json")
    p.add_argument("--layout", type=Path, default=ROOT / "data" / "layout.json")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "digits.json")
    args = p.parse_args()

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"ERROR: no such video: {video}")
    rows = json.loads(args.labels.read_text(encoding="utf-8"))["frames"]
    anchors = json.loads(args.layout.read_text(encoding="utf-8"))["enemy_hp_text"]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"ERROR: OpenCV could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    train = [r for r in rows if r["use"] == "train"]
    test = [r for r in rows if r["use"] != "train"]
    print(f"  {len(train)} training frames, {len(test)} held out")

    tmpl, adv, _ = assemble(harvest(cap, fps, train, anchors))
    first = DigitReader({d: v * 255 for d, v in tmpl.items()}, adv)
    offsets = realign(first, cap, fps, train, anchors)
    tmpl, adv, counts = assemble(harvest(cap, fps, train, anchors, offsets))

    missing = [d for d in DIGITS if d not in tmpl]
    if missing:
        sys.exit(f"ERROR: no sample of {', '.join(missing)} in the training "
                 "frames. Add a label whose value contains them.")
    print("  glyphs per digit: " + " ".join(f"{d}:{counts[d]}" for d in DIGITS))

    reader = DigitReader({d: v * 255 for d, v in tmpl.items()}, adv)
    reader.save(args.out, note=f"Cut from {video.name} using "
                               f"{args.labels.name}; see tools/build_digits.py")
    print(f"  wrote {args.out}")

    print("\n  === held out: numbers the templates never saw ===")
    tally = defaultdict(lambda: [0, 0, 0])      # seg -> [right, wrong, blank]
    for row in test:
        for side, x0 in anchors.items():
            want = row[side.lower()]
            got, score = reader.read(strip_at(cap, fps, row["t"], x0))
            bucket = tally[row.get("seg", "all")]
            if got is None:
                bucket[2] += 1
                flag = "unread"
            elif str(got) == want:
                bucket[0] += 1
                flag = "ok"
            else:
                bucket[1] += 1
                flag = "WRONG"
            if flag != "ok":
                print(f"    {row['t']:>4}s {side:<5s} want {want:>4s} "
                      f"got {str(got):>5s} ({score:.2f}) {flag}")
    print()
    for seg, (right, wrong, blank) in sorted(tally.items()):
        n = right + wrong + blank
        print(f"  {seg:<6s} exact {right}/{n} = {100 * right / n:.0f}%   "
              f"wrong {wrong}   unread {blank}")
    cap.release()


if __name__ == "__main__":
    main()
