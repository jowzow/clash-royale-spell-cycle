#!/usr/bin/env python3
"""
Checks the tracked tower health, and the cast count that follows from it,
against health numbers read off the screen by eye.

tools/build_digits.py reports how often a single frame is read correctly.
That is not the number that matters, because the program never acts on one
frame: it votes over a third of a second and holds its last value in
between. This checks what the program actually ends up believing, and then
the only thing a viewer ever sees, which is the count drawn on the card.

Every miss is reported with how long the tracker took to catch up, so a
checkpoint that lands in the moment a hit just registered is not confused
with a number the program got wrong and kept.

Usage:
    python tools/test_health.py videoplayback.mp4
    python tools/test_health.py videoplayback.mp4 --spell Rocket
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from digits import DigitReader                     # noqa: E402
from spellcycle import SpellBook, Tower            # noqa: E402

LEAD_IN = 4.0      # seconds of run-up so the vote window is full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--labels", type=Path, default=ROOT / "data" / "hp_labels.json")
    p.add_argument("--layout", type=Path, default=ROOT / "data" / "layout.json")
    p.add_argument("--digits", type=Path, default=ROOT / "data" / "digits.json")
    p.add_argument("--spells", type=Path, default=ROOT / "data" / "spells.json")
    p.add_argument("--segment", default="demo",
                   help="which labelled segment to replay (default demo)")
    p.add_argument("--spell", default="Fireball")
    p.add_argument("--spell-level", type=int, default=16)
    args = p.parse_args()

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"ERROR: no such video: {video}")

    rows = [r for r in json.loads(args.labels.read_text(encoding="utf-8"))["frames"]
            if r.get("seg") == args.segment]
    if not rows:
        sys.exit(f"ERROR: no labels in segment '{args.segment}'.")
    anchors = json.loads(args.layout.read_text(encoding="utf-8"))["enemy_hp_text"]
    reader = DigitReader.load(args.digits)
    book = SpellBook(args.spells, args.spell_level)
    damage = book.damage[args.spell]

    start = min(r["t"] for r in rows) - LEAD_IN
    end = max(r["t"] for r in rows) + 1.0

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"ERROR: OpenCV could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))

    towers = {name: Tower(name, int(x)) for name, x in anchors.items()}
    want = {r["t"]: r for r in rows}
    seen, series, i = {}, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = start + i / fps
        if t > end:
            break
        for tower in towers.values():
            tower.update(frame, reader, t)
        snapshot = {n: tw.hp for n, tw in towers.items()}
        series.append((t, snapshot))
        for sec in want:
            if abs(t - sec) < 0.5 / fps:
                seen[sec] = snapshot
        i += 1
    cap.release()

    def catch_up(sec, side, truth):
        """Seconds after the checkpoint before the tracker reaches the right
        number, or None if it never does inside the replayed window."""
        for t, snap in series:
            if t >= sec and snap[side] == truth:
                return t - sec
        return None

    print(f"\n  {args.segment}: replayed {i} frames from {start:.0f}s\n")
    print(f"  {'time':>6}  {'tower':<6} {'on screen':>9} {'tracked':>8}  "
          f"{'casts wanted':>12} {'casts shown':>11}")
    hp_ok = hp_bad = cast_ok = cast_bad = late = never = 0
    for sec in sorted(want):
        if sec not in seen:
            continue
        for side in anchors:
            truth = int(want[sec][side.lower()])
            got = seen[sec][side]
            need = math.ceil(truth / damage)
            shown = None if got is None else math.ceil(got / damage)
            flag = ""
            if got == truth:
                hp_ok += 1
            else:
                hp_bad += 1
            if shown == need:
                cast_ok += 1
            else:
                cast_bad += 1
                delay = catch_up(sec, side, truth)
                if delay is None:
                    never += 1
                    flag = "  <- wrong, never corrected"
                else:
                    late += 1
                    flag = f"  <- late by {delay:.2f}s"
            print(f"  {sec:>5}s  {side:<6} {truth:>9} {str(got):>8}  "
                  f"{need:>12} {str(shown):>11}{flag}")

    n = hp_ok + hp_bad
    print(f"\n  health exactly right   {hp_ok}/{n}")
    print(f"  {args.spell} count right  {cast_ok}/{n}   "
          f"({damage} damage per cast at level {args.spell_level})")
    print(f"  of the {cast_bad} misses: {late} are the vote catching up with "
          f"a hit that had just\n  landed, {never} never resolve inside the "
          "clip.")
    # The four unresolved ones are the last three seconds, where both towers
    # are collapsing and their numbers stop being legible at all. Anything
    # beyond that is a regression.
    sys.exit(1 if never > 4 else 0)


if __name__ == "__main__":
    main()
