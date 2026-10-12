#!/usr/bin/env python3
"""
Builds the YOLO training set for the spell detector from a gameplay video.

Hand labelling thousands of frames was not an option, so the labels come from
a two-stage trick:

1. A spell IMPACT is a huge bright blast, far larger than any gold badge or
   fire-effect troop in the arena. A strict area threshold finds those
   reliably, and gives one anchor per cast.
2. A spell IN FLIGHT is a small orange ball that is NOT reliably separable by
   colour alone. But it has to arrive where the blast happens, so for each
   anchor only the ~1 second before it is searched, and only near the impact
   point. That turns an impossible global search into an easy local one.

The model then learns the projectile's appearance so it can be found without
knowing where it is going to land, which is the whole point: a fixed region
or a colour rule cannot do that, a detector can.

Usage:
    python tools/build_dataset.py videoplayback.mp4
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ARENA = (0.10, 0.78)          # vertical slice holding the playfield
IMPACT_MIN_AREA = 1500        # a blast; gold badges never get near this
FLIGHT_AREA = (25, 900)       # the ball itself
FLIGHT_LOOKBACK = 30          # frames before impact to search
FLIGHT_RADIUS = 150           # px from the impact point
CLASSES = ["spell_in_flight", "spell_impact"]

LO = np.array([3, 120, 190])
HI = np.array([30, 255, 255])

# The arena is full of permanently orange furniture: the two bridge gates, the
# king tower crowns, every gold level badge. Colour alone cannot tell those
# from a fireball, and the first version of this script happily labelled
# bridges as projectiles. The fix is to learn where orange is ALWAYS present
# and refuse to label anything there, since a spell is by definition transient.
STATIC_MAX = 0.12


def static_prior(video, cache: Path):
    """Per-pixel fraction of sampled frames in which that pixel is orange."""
    if cache.exists():
        return np.load(cache)
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc, n = None, 0
    for i in range(0, total, 53):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        m = cv2.inRange(cv2.cvtColor(fr, cv2.COLOR_BGR2HSV), LO, HI) > 0
        acc = m.astype(np.float32) if acc is None else acc + m
        n += 1
    cap.release()
    prior = acc / max(n, 1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, prior)
    print(f"    static-orange prior built from {n} frames")
    return prior


def blobs(frame, y0, prior=None):
    hsv = cv2.cvtColor(frame[y0:], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LO, HI)
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        y += y0
        if prior is not None and prior[y:y + h, x:x + w].mean() > STATIC_MAX:
            continue      # furniture, not a spell
        out.append((a, x, y, w, h, cent[i][0], cent[i][1] + y0))
    return out


def find_events(cap, y0, total, prior):
    """Frame ranges where a blast is on screen, with its centre."""
    raw = {}
    for i in range(total):
        ok, fr = cap.read()
        if not ok:
            break
        big = [b for b in blobs(fr, y0, prior) if b[0] >= IMPACT_MIN_AREA]
        if big:
            raw[i] = max(big)
    frames = sorted(raw)
    if not frames:
        return []
    events, cur = [], [frames[0]]
    for f in frames[1:]:
        if f - cur[-1] <= 15:
            cur.append(f)
        else:
            events.append(cur)
            cur = [f]
    events.append(cur)
    return [(e, raw[max(e, key=lambda f: raw[f][0])]) for e in events]


def write(split, idx, frame, boxes, root):
    h, w = frame.shape[:2]
    stem = f"{split}_{idx:05d}"
    cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), frame,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(root / "labels" / split / f"{stem}.txt", "w") as fh:
        for cls, (x, y, bw, bh) in boxes:
            fh.write(f"{cls} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                     f"{bw / w:.6f} {bh / h:.6f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out", type=Path, default=Path("dataset"))
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    y0 = int(ARENA[0] * h)

    print("  pass 0: learning where orange is permanent")
    prior = static_prior(args.video, Path("output/orange_prior.npy"))

    print("  pass 1: finding impacts")
    events = find_events(cap, y0, total, prior)
    print(f"    {len(events)} impact events")

    # Hold out whole events, never single frames: consecutive frames of one
    # cast are near duplicates and would leak straight into validation.
    rng = random.Random(args.seed)
    order = list(range(len(events)))
    rng.shuffle(order)
    n_val = max(1, int(len(events) * args.val_frac))
    val_events = set(order[:n_val])
    print(f"    {len(events) - n_val} train events, {n_val} val events")

    root = args.out
    if root.exists():
        shutil.rmtree(root)
    for s in ("train", "val"):
        (root / "images" / s).mkdir(parents=True)
        (root / "labels" / s).mkdir(parents=True)

    wanted = {}
    for ei, (frames, peak) in enumerate(events):
        split = "val" if ei in val_events else "train"
        cx, cy = peak[5], peak[6]
        for f in frames:
            wanted.setdefault(f, []).append((ei, split, "impact", cx, cy))
        for f in range(max(0, frames[0] - FLIGHT_LOOKBACK), frames[0]):
            wanted.setdefault(f, []).append((ei, split, "flight", cx, cy))

    busy = set()
    for frames, _ in events:
        busy.update(range(frames[0] - 90, frames[-1] + 90))
    free = [f for f in range(0, total, 37) if f not in busy]
    rng.shuffle(free)
    for f in free[:400]:
        wanted.setdefault(f, []).append((-1, "train" if rng.random() > .25
                                         else "val", "negative", 0, 0))

    print("  pass 2: writing frames")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    counts = {"spell_impact": 0, "spell_in_flight": 0, "negative": 0}
    idx = {"train": 0, "val": 0}
    for i in range(total):
        ok, fr = cap.read()
        if not ok:
            break
        if i not in wanted:
            continue
        for ei, split, kind, cx, cy in wanted[i]:
            found = blobs(fr, y0, prior)
            boxes = []
            if kind == "impact":
                big = [b for b in found if b[0] >= IMPACT_MIN_AREA]
                if big:
                    a, x, y, bw, bh, _, _ = max(big)
                    boxes.append((1, (x, y, bw, bh)))
            elif kind == "flight":
                near = [b for b in found
                        if FLIGHT_AREA[0] <= b[0] <= FLIGHT_AREA[1]
                        and np.hypot(b[5] - cx, b[6] - cy) <= FLIGHT_RADIUS]
                if near:
                    a, x, y, bw, bh, _, _ = max(near)
                    boxes.append((0, (x, y, bw, bh)))
            if kind != "negative" and not boxes:
                continue
            write(split, idx[split], fr, boxes, root)
            idx[split] += 1
            counts[{"impact": "spell_impact", "flight": "spell_in_flight",
                    "negative": "negative"}[kind]] += 1

    (root / "data.yaml").write_text(
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\n\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n", encoding="utf-8")

    print("\n  dataset written to", root)
    for k, v in counts.items():
        print(f"    {k:16s} {v:5d}")
    print(f"    train images     {idx['train']:5d}")
    print(f"    val images       {idx['val']:5d}")


if __name__ == "__main__":
    main()
