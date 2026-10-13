#!/usr/bin/env python3
"""
SPELL CYCLE COUNTER - how many casts to delete the enemy's weakest tower.

Watches a recorded Clash Royale match, reads the enemy princess towers'
health straight off the numbers the game prints above their bars, finds
which spells are sitting in your hand, and draws a number on each one: how
many casts of that spell it takes to finish the weakest tower off.

Spell damage to a crown tower is guaranteed, so unlike a push the count is
exact. That is the whole idea behind spell cycling.

Usage:
    python spellcycle.py videoplayback.mp4 --start 636 --end 690 --spell-level 16
    python spellcycle.py match.mp4 --spell-level 11
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from digits import BAND, SPAN, DigitReader

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Spell maths
# --------------------------------------------------------------------------


class SpellBook:
    """Crown tower damage per spell, and how many casts a tower costs."""

    def __init__(self, path: Path, level: int):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            sys.exit(f"ERROR: {path} is missing. Run:\n"
                     "    python tools/build_spell_table.py")
        except json.JSONDecodeError as exc:
            sys.exit(f"ERROR: {path} is not valid JSON: {exc}")

        self.level = level
        self.spells = blob["spells"]
        self.generated = blob.get("_generated", "unknown")

        self.damage = {}
        for name, info in self.spells.items():
            table = info["crown_tower_damage"]
            if str(level) in table:
                self.damage[name] = table[str(level)]
            else:
                # Below a card's minimum level (Legendaries start at 9), so
                # clamp to the nearest level that exists rather than crash.
                levels = sorted(int(k) for k in table)
                near = min(levels, key=lambda L: abs(L - level))
                self.damage[name] = table[str(near)]
                print(f"  note: {name} has no level {level}; using level "
                      f"{near} ({self.damage[name]} damage)")

    def casts_needed(self, spell: str, hp) -> int | None:
        """Casts to take `hp` to zero. None if the spell or the health is
        unknown, or the tower is already down."""
        dmg = self.damage.get(spell)
        if not dmg or hp is None or hp <= 0:
            return None
        return math.ceil(hp / dmg)


# --------------------------------------------------------------------------
# Tracking a tower's health
# --------------------------------------------------------------------------


class Tower:
    """One enemy princess tower, tracked by reading its printed health.

    Any one frame can be misread: a troop walks over the number, the game
    animates it rolling down after a hit, or the blur on a 7 pixel glyph
    simply wins. So no single frame is trusted. A value becomes the tower's
    health only once it is what most of the last half second says, and a
    reading that claims the tower healed needs a near unanimous window
    before it is believed, because towers do not heal.

    Between confirmations the last confirmed value stands. That is the
    fallback: the number can vanish behind a spell animation for a second
    and the count on the cards does not flinch.
    """

    WINDOW = 9           # frames considered, about a third of a second
    AGREE = 6            # how many must say the same number
    SURE = 8             # ...and how many for a reading that looks wrong

    def __init__(self, name: str, x: int):
        self.name = name
        self.x = x
        self.hp = None
        self.alive = True
        self.history = deque(maxlen=300)     # (time, hp), for scoring impacts
        self._recent = deque(maxlen=self.WINDOW)
        self._blank = 0

    def update(self, frame, reader: DigitReader, t: float):
        strip = frame[BAND[0]:BAND[1], self.x:self.x + SPAN]
        value, _score = reader.read(strip)
        self._recent.append(value)

        if value is None:
            # A destroyed tower's health number is gone for good. Nothing
            # covers it for four straight seconds, so a gap that long means
            # the tower is down rather than hidden.
            self._blank += 1
            if self._blank > 120 and self.hp is not None:
                self.alive = False
        else:
            self._blank = 0

        counts = Counter(v for v in self._recent if v is not None)
        if counts:
            best, votes = counts.most_common(1)[0]
            jumped = self.hp is not None and (
                best > self.hp or abs(best - self.hp) > 0.4 * self.hp)
            if votes >= (self.SURE if jumped else self.AGREE):
                self.hp = best
                if best <= 0:
                    self.alive = False
        self.history.append((t, self.hp))


# --------------------------------------------------------------------------
# Watching the arena with the trained detector
# --------------------------------------------------------------------------


class SpellWatcher:
    """Finds spells in the arena using the fine-tuned YOLOv8n detector.

    This is the one part of the program a fixed rectangle cannot do. The hand
    and the health numbers never move, so template matching and a fixed strip
    are the right tools there. A spell in flight has no fixed position, no
    fixed scale, and crosses whatever happens to be on the board, so it gets
    a detector trained on it. See tools/train_model.py.

    Catching the impact also lets the program check its own arithmetic: when
    a blast lands near a tower, the health drop that follows is compared with
    the per-cast damage the counter was promising.
    """

    FLIGHT, IMPACT = "spell_in_flight", "spell_impact"

    def __init__(self, weights: Path, conf: float):
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("ERROR: ultralytics is not installed but the detector "
                     "was asked for. pip install -r requirements.txt, or "
                     "pass --no-model.")
        os.environ.setdefault("YOLO_VERBOSE", "False")
        self.model = YOLO(str(weights))
        self.conf = conf
        self.names = self.model.names

    def detect(self, frame):
        """-> list of (class_name, (x1, y1, x2, y2), confidence)."""
        res = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            xy = [int(v) for v in b.xyxy[0].tolist()]
            out.append((self.names[int(b.cls)], tuple(xy), float(b.conf)))
        return out


# --------------------------------------------------------------------------
# Reading the hand
# --------------------------------------------------------------------------


class HandReader:
    """Identifies which of the four hand slots hold a damaging spell.

    The four slots sit at fixed places on screen, so each slot is matched
    against every spell's card art rather than sliding templates over the
    whole frame. That is faster and it cannot put a card in the wrong place.
    """

    def __init__(self, cards_dir: Path, names, threshold: float):
        self.threshold = threshold
        self.art = {}
        for name in names:
            key = name.lower().replace(" ", "-")
            img = cv2.imread(str(cards_dir / f"{key}.png"), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"  warning: no card art for {name}")
                continue
            if img.ndim > 2 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # Crop to the inner art. The rarity frame and elixir badge differ
            # between the downloaded asset and the in-game hand; the art does
            # not.
            h, w = img.shape[:2]
            self.art[name] = img[int(h * 0.18):int(h * 0.80),
                                 int(w * 0.14):int(w * 0.86)]
        if not self.art:
            sys.exit("ERROR: no card art found. Expected PNGs in cards/.")

    def read(self, frame, slots):
        """-> list of (slot_index, spell, box, score) for slots holding spells."""
        h, w = frame.shape[:2]
        found = []
        for i, slot in enumerate(slots):
            x1, y1 = int(slot[0] * w), int(slot[1] * h)
            x2, y2 = int(slot[2] * w), int(slot[3] * h)
            cell = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if cell.size == 0 or cell.shape[0] < 12 or cell.shape[1] < 12:
                continue

            # Template a little smaller than the slot leaves a few pixels of
            # search room, which absorbs the card's bob animation.
            th = max(8, int(cell.shape[0] * 0.62))
            tw = max(8, int(cell.shape[1] * 0.74))

            best = (0.0, None, None)
            for name, art in self.art.items():
                small = cv2.resize(art, (tw, th), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(cell, small, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(res)
                if score > best[0]:
                    best = (score, name, loc)

            score, name, loc = best
            if score >= self.threshold and name:
                found.append((i, name,
                              (x1 + loc[0], y1 + loc[1],
                               x1 + loc[0] + tw, y1 + loc[1] + th), score))
        return found


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_DUPLEX
GOLD = (60, 200, 255)
DET_COLOURS = {"spell_in_flight": (255, 190, 60), "spell_impact": (80, 90, 255)}


def draw_detection(frame, name, box, conf, s):
    x1, y1, x2, y2 = box
    col = DET_COLOURS.get(name, (200, 200, 200))
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, max(1, int(2 * s)))
    tag = f"{'IN FLIGHT' if name == 'spell_in_flight' else 'IMPACT'} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(tag, FONT, 0.34 * s, 1)
    ty = max(th + int(3 * s), y1 - int(3 * s))
    cv2.rectangle(frame, (x1, ty - th - int(3 * s)),
                  (x1 + tw + int(5 * s), ty + int(2 * s)), col, -1)
    cv2.putText(frame, tag, (x1 + int(2 * s), ty), FONT, 0.34 * s,
                (20, 18, 16), 1, cv2.LINE_AA)


def draw_spell(frame, box, spell, casts, s):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), GOLD, max(1, int(2 * s)))

    label = "-" if casts is None else str(casts)
    scale = 0.95 * s
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, 2)
    cx = (x1 + x2) // 2
    by = max(th + int(6 * s), y1 - int(5 * s))
    pad = int(5 * s)

    cv2.rectangle(frame, (cx - tw // 2 - pad, by - th - pad),
                  (cx + tw // 2 + pad, by + pad // 2), (18, 16, 14), -1)
    cv2.rectangle(frame, (cx - tw // 2 - pad, by - th - pad),
                  (cx + tw // 2 + pad, by + pad // 2), GOLD, max(1, int(s)))
    cv2.putText(frame, label, (cx - tw // 2, by), FONT, scale, GOLD,
                max(1, int(2 * s)), cv2.LINE_AA)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def load_json(path: Path, hint: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"ERROR: {path} is missing.\n{hint}")
    except json.JSONDecodeError as exc:
        sys.exit(f"ERROR: {path} is not valid JSON: {exc}")


def main():
    p = argparse.ArgumentParser(
        description="Count spell casts needed to delete the weakest enemy tower.")
    p.add_argument("video")
    p.add_argument("--spell-level", type=int, default=11,
                   help="level of your spells (default 11, tournament "
                        "standard). The game prints the level under a card's "
                        "name every time it is played, so you can read it off "
                        "the clip; videoplayback.mp4 is level 16.")
    p.add_argument("--start", type=float, default=0, help="start second")
    p.add_argument("--end", type=float, default=0, help="end second (0 = all)")
    p.add_argument("--layout", type=Path, default=ROOT / "data" / "layout.json")
    p.add_argument("--spells", type=Path, default=ROOT / "data" / "spells.json")
    p.add_argument("--digits", type=Path, default=ROOT / "data" / "digits.json")
    p.add_argument("--cards", type=Path, default=ROOT / "cards")
    p.add_argument("--threshold", type=float, default=0.78,
                   help="card match score to accept a spell (default 0.78). A "
                        "real spell scores about 0.86-0.92; the best false "
                        "positive seen was a greyed-out post-match card at 0.72.")
    p.add_argument("--model", type=Path, default=ROOT / "models" / "spell_yolov8n.pt",
                   help="trained spell detector weights")
    p.add_argument("--spell-conf", type=float, default=0.35,
                   help="detector confidence threshold (default 0.35)")
    p.add_argument("--no-model", action="store_true",
                   help="skip the arena detector, hand and health only")
    p.add_argument("--every", type=int, default=5,
                   help="re-read the hand and run the detector every N frames "
                        "(default 5). Health is read every frame.")
    p.add_argument("--outdir", type=Path, default=ROOT / "output")
    args = p.parse_args()

    if not 1 <= args.spell_level <= 16:
        sys.exit("ERROR: --spell-level must be between 1 and 16.")

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"ERROR: no such video: {video}")

    layout = load_json(args.layout,
                       "It holds the on-screen regions for your recording. Run\n"
                       "    python tools/inspect_frame.py <video> --grid\n"
                       "and edit a copy of data/layout.example.json to match.")
    if "enemy_hp_text" not in layout:
        sys.exit(f"ERROR: {args.layout} has no 'enemy_hp_text' section. It "
                 "names the left edge of each enemy tower's health number.")
    if not args.digits.exists():
        sys.exit(f"ERROR: {args.digits} is missing. Build the glyph templates:\n"
                 f"    python tools/build_digits.py {video}")

    book = SpellBook(args.spells, args.spell_level)
    reader = DigitReader.load(args.digits)
    hand_reader = HandReader(args.cards, list(book.spells), args.threshold)
    args.outdir.mkdir(parents=True, exist_ok=True)

    watcher = None
    if not args.no_model:
        if args.model.exists():
            watcher = SpellWatcher(args.model, args.spell_conf)
        else:
            print(f"  note: no detector at {args.model}, arena watching off.\n"
                  "        Train one with:\n"
                  "            python tools/build_dataset.py <video>\n"
                  "            python tools/train_model.py")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"ERROR: OpenCV could not open {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = max(0.7, w / 298.0)

    towers = [Tower(name, int(x)) for name, x in layout["enemy_hp_text"].items()]
    slots = layout["card_slots"]

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    stop_at = args.end if args.end > 0 else None

    out_path = args.outdir / f"{video.stem}_spellcycle.mp4"
    # Opened on the first frame actually read, so a run that reads nothing
    # (a --start past the end, say) leaves any previous output untouched
    # instead of truncating it.
    writer = None

    print(f"\n  {video.name}: {w}x{h} @ {fps:.0f} fps")
    print(f"  Spells at level {args.spell_level}")
    print("  Crown tower damage: "
          + ", ".join(f"{n} {d}" for n, d in sorted(book.damage.items())))
    print(f"  Spell table generated {book.generated}\n")

    frames = 0
    hand = []
    dets = []
    last_line = None
    impact_on = False
    pending = []
    casts_seen = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = args.start + frames / fps
            if stop_at and t >= stop_at:
                break
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (frame.shape[1], frame.shape[0]))

            for tower in towers:
                tower.update(frame, reader, t)
            known = [tw for tw in towers if tw.alive and tw.hp is not None]
            weakest = min(known, key=lambda tw: tw.hp) if known else None

            if frames % args.every == 0:
                hand = hand_reader.read(frame, slots)
                if watcher:
                    dets = watcher.detect(frame)

            for name, box, conf in dets:
                draw_detection(frame, name, box, conf, s)

            # A blast is one cast. Record the tower's health just before it
            # and again once the damage has registered, so the program can
            # check the number it was promising against what actually landed.
            # Only blasts on the enemy half count: a spell landing on your own
            # tower is still a spell, but it says nothing about the number
            # this program is drawing on the enemy's health.
            hit = [d for d in dets if d[0] == SpellWatcher.IMPACT
                   and (d[1][1] + d[1][3]) / 2 < h * 0.47]
            if hit and not impact_on:
                impact_on = True
                casts_seen += 1
                bx = (hit[0][1][0] + hit[0][1][2]) / 2
                side = towers[0] if bx < w / 2 else towers[-1]
                before = next((hp for ts, hp in reversed(side.history)
                               if ts <= t - 0.4 and hp is not None), side.hp)
                pending.append([t, side, before])
                print(f"  [{t:7.2f}s] spell impact #{casts_seen} on the "
                      f"{side.name.lower()} tower")
            elif not hit:
                impact_on = False

            for item in pending[:]:
                if t - item[0] >= 1.6:
                    _, side, before = item
                    if before is None or side.hp is None:
                        print(f"  [{t:7.2f}s]   not measurable, that tower's "
                              "health was not readable")
                    elif not side.alive:
                        print(f"  [{t:7.2f}s]   {side.name.lower()} tower "
                              "went down")
                    else:
                        # Deliberately not called "the spell's damage": this
                        # is every point the tower lost in the window, so a
                        # troop hitting it at the same time is counted too.
                        print(f"  [{t:7.2f}s]   {side.name.lower()} tower "
                              f"{before} -> {side.hp}, lost "
                              f"{before - side.hp} HP over 1.6s")
                    pending.remove(item)

            for _slot, spell, box, _score in hand:
                casts = book.casts_needed(spell, weakest.hp) if weakest else None
                draw_spell(frame, box, spell, casts, s)

            writer.write(frame)

            if weakest and hand:
                line = (f"weakest {weakest.name.lower()} {weakest.hp} HP:  "
                        + ",  ".join(f"{sp} x{book.casts_needed(sp, weakest.hp)}"
                                     for _i, sp, _b, _sc in hand))
                if line != last_line:
                    print(f"  [{t:7.2f}s] {line}")
                    last_line = line

            frames += 1
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frames == 0:
        sys.exit("ERROR: no frames were read. Check --start against the "
                 "video length.")

    print(f"\n  Saved {out_path}  ({frames} frames)")
    if last_line is None:
        print("  Nothing was counted. Either no spell matched in the hand "
              "(lower --threshold, or check card_slots in the layout file) or "
              "no tower's health could be read (check enemy_hp_text, and "
              "rebuild the glyphs with tools/build_digits.py).")


if __name__ == "__main__":
    main()
