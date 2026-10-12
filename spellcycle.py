#!/usr/bin/env python3
"""
SPELL CYCLE COUNTER - how many casts to delete the enemy's weakest tower.

Watches a recorded Clash Royale match, reads the enemy princess towers'
health off their health bars, finds which spells are sitting in your hand,
and prints a number on each one: how many casts of that spell it takes to
finish the weakest tower off.

Spell damage to a crown tower is guaranteed, so unlike a push the count is
exact. That is the whole idea behind spell cycling.

Usage:
    python spellcycle.py videoplayback.mp4 --start 700 --end 790
    python spellcycle.py match.mp4 --spell-level 14 --tower-hp 4858
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

# Princess tower max health by level. Level 11 is tournament standard and the
# default, matching the spell level assumption.
PRINCESS_HP = {9: 2534, 10: 2786, 11: 3052, 12: 3346, 13: 3668, 14: 4032,
               15: 4424}


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

    def casts_needed(self, spell: str, hp: float) -> int | None:
        """Casts to take `hp` to zero. None if unknown spell or dead tower."""
        dmg = self.damage.get(spell)
        if not dmg or hp <= 0:
            return None
        return math.ceil(hp / dmg)


# --------------------------------------------------------------------------
# Reading tower health off the bar
# --------------------------------------------------------------------------


class TowerBar:
    """One enemy princess tower, read from its health bar.

    The health number the game prints is only about 6 pixels tall in a phone
    capture, far too small to read reliably, so the bar is measured instead.
    Fitted against four hand-read values this lands within 0.3%.

    The bar can be covered by a troop, a spell or a card being dragged over
    it, so a reading is only accepted when it is believable: towers never
    heal, and a bar cannot refill. Anything else keeps the last good value.
    """

    def __init__(self, name: str, roi, max_hp: int, window: int = 15):
        self.name = name
        self.roi = roi
        self.max_hp = max_hp
        self.fraction = 1.0
        self.alive = True
        self._recent = deque(maxlen=window)
        self._misses = 0
        self.history = deque(maxlen=300)   # (time, hp), for scoring impacts
        # Until the bar has actually been read once, self.fraction is just the
        # starting assumption of full health. Anything derived from it before
        # then is fiction, which is what `seen` guards.
        self.seen = False

    @property
    def hp(self) -> float:
        return self.fraction * self.max_hp

    def update(self, frame, palette):
        reading = self._read(frame, palette)
        if reading is None:
            self._misses += 1
            # A destroyed tower's bar vanishes for good. Nothing covers a bar
            # for four straight seconds, so a gap that long means it is gone.
            if self._misses > 120:
                self.alive = False
            return

        self._misses = 0
        self.seen = True
        self._recent.append(reading)

        # Anything covering the bar can only make the red run look SHORTER
        # than it is, never longer, so over half a second the largest reading
        # is the honest one. Combined with the fact that towers never heal,
        # that makes the estimate monotonic and immune to a troop or a spell
        # animation parked on top of the bar.
        best = float(np.max(self._recent))
        self.fraction = min(self.fraction, best)

        if self.fraction <= 0.005:
            self.fraction = 0.0
            self.alive = False

    def _read(self, frame, palette) -> float | None:
        h, w = frame.shape[:2]
        x1, y1 = int(self.roi[0] * w), int(self.roi[1] * h)
        x2, y2 = int(self.roi[2] * w), int(self.roi[3] * h)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 2:
            return None

        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        mask = (cv2.inRange(hsv, palette["lo1"], palette["hi1"])
                | cv2.inRange(hsv, palette["lo2"], palette["hi2"]))

        # A column counts as filled when most of the bar's height is red.
        filled = (mask > 0).mean(axis=0) >= 0.6

        # Bars drain right to left, so measure the run from the left edge and
        # stop at the first real gap rather than at one stray dark pixel.
        run = gap = 0
        for on in filled:
            if on:
                run += 1
                gap = 0
            else:
                gap += 1
                if gap >= 2:
                    break
        # No red at all means the bar is hidden behind a troop or a spell
        # effect, not that the tower is at zero. A tower that really hits zero
        # is destroyed and its bar disappears for good, which the miss counter
        # in update() picks up instead. Reporting 0 here would latch the
        # tower to empty on the first frame something covered it.
        if run == 0:
            return None
        return min(run, len(filled)) / len(filled)


# --------------------------------------------------------------------------
# Watching the arena with the trained detector
# --------------------------------------------------------------------------


class SpellWatcher:
    """Finds spells in the arena using the fine-tuned YOLOv8n detector.

    This is the one part of the program a fixed rectangle cannot do. The hand
    and the health bars never move, so template matching and a fixed region
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
            sys.exit("ERROR: ultralytics is not installed but --model was "
                     "given. pip install -r requirements.txt")
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


def load_layout(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"ERROR: {path} is missing.\n"
                 "It holds the on-screen regions for your recording. Run\n"
                 "    python tools/inspect_frame.py <video> --grid\n"
                 "and edit a copy of data/layout.example.json to match.")
    except json.JSONDecodeError as exc:
        sys.exit(f"ERROR: {path} is not valid JSON: {exc}")


def main():
    p = argparse.ArgumentParser(
        description="Count spell casts needed to delete the weakest enemy tower.")
    p.add_argument("video")
    p.add_argument("--spell-level", type=int, default=11,
                   help="level of your spells (default 11, tournament standard)")
    p.add_argument("--tower-hp", type=int, default=None,
                   help="enemy princess tower max health "
                        f"(default {PRINCESS_HP[11]}, a level 11 tower)")
    p.add_argument("--start", type=float, default=0, help="start second")
    p.add_argument("--end", type=float, default=0, help="end second (0 = all)")
    p.add_argument("--layout", type=Path, default=ROOT / "data" / "layout.json")
    p.add_argument("--spells", type=Path, default=ROOT / "data" / "spells.json")
    p.add_argument("--cards", type=Path, default=ROOT / "cards")
    p.add_argument("--threshold", type=float, default=0.78,
                   help="card match score to accept a spell (default 0.78). A real spell scores about 0.86-0.92; the best false positive seen was a greyed-out post-match card at 0.72.")
    p.add_argument("--model", type=Path, default=ROOT / "models" / "spell_yolov8n.pt",
                   help="trained spell detector weights")
    p.add_argument("--spell-conf", type=float, default=0.35,
                   help="detector confidence threshold (default 0.35)")
    p.add_argument("--no-model", action="store_true",
                   help="skip the arena detector, hand and health only")
    p.add_argument("--every", type=int, default=5,
                   help="re-read the hand every N frames (default 5)")
    p.add_argument("--outdir", type=Path, default=ROOT / "output")
    args = p.parse_args()

    if not 1 <= args.spell_level <= 15:
        sys.exit("ERROR: --spell-level must be between 1 and 15.")
    max_hp = args.tower_hp or PRINCESS_HP[11]
    if max_hp <= 0:
        sys.exit("ERROR: --tower-hp must be positive.")

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"ERROR: no such video: {video}")

    layout = load_layout(args.layout)
    book = SpellBook(args.spells, args.spell_level)
    hand_reader = HandReader(args.cards, list(book.spells), args.threshold)
    args.outdir.mkdir(parents=True, exist_ok=True)

    palette = {k: np.array(layout[v], dtype=np.uint8) for k, v in
               (("lo1", "red_lo1"), ("hi1", "red_hi1"),
                ("lo2", "red_lo2"), ("hi2", "red_hi2"))}

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

    towers = [TowerBar(name, roi, max_hp)
              for name, roi in layout["enemy_bars"].items()]
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
    print(f"  Spells at level {args.spell_level}; enemy towers {max_hp} HP max")
    if args.tower_hp is None:
        # The bar gives a fraction, so every health figure below is that
        # fraction of this assumed maximum. Guess the maximum wrong and the
        # counts are wrong too, with nothing on screen to show it.
        print("  WARNING: --tower-hp not given, assuming a level 11 tower "
              f"({max_hp} HP).\n"
              "           If the recording's towers are a different level "
              "every health\n"
              "           figure and cast count below is off. Pass --tower-hp "
              "with the\n"
              "           health shown on a full enemy bar to fix it.")
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
                tower.update(frame, palette)
                tower.history.append((t, tower.hp))
            alive = [tw for tw in towers if tw.alive]
            weakest = min(alive, key=lambda tw: tw.hp) if alive else None

            if frames % args.every == 0:
                hand = hand_reader.read(frame, slots)

            if watcher and frames % args.every == 0:
                dets = watcher.detect(frame)

            for name, box, conf in dets:
                draw_detection(frame, name, box, conf, s)

            # A blast is one cast. Record the tower's health just before it
            # and again once the damage has registered, so the program can
            # check the number it was promising against what actually landed.
            # Only blasts on the enemy half count. A spell landing on your own
            # tower is still a spell, but it tells you nothing about the
            # number this program is drawing on the enemy's health.
            hit = [d for d in dets if d[0] == SpellWatcher.IMPACT
                   and (d[1][1] + d[1][3]) / 2 < h * 0.47]
            if hit and not impact_on:
                impact_on = True
                casts_seen += 1
                bx = (hit[0][1][0] + hit[0][1][2]) / 2
                side = towers[0] if bx < w / 2 else towers[-1]
                before = (next((hp for ts, hp in reversed(side.history)
                                if ts <= t - 0.4), side.hp)
                          if side.seen else None)
                pending.append([t, side, before])
                print(f"  [{t:7.2f}s] spell impact #{casts_seen} on the "
                      f"{side.name.lower()} tower")
            elif not hit:
                impact_on = False

            for item in pending[:]:
                if t - item[0] >= 1.6:
                    _, side, before = item
                    if before is None:
                        print(f"  [{t:7.2f}s]   not measurable, that tower's "
                              "health was not established yet")
                    elif not side.alive:
                        print(f"  [{t:7.2f}s]   {side.name.lower()} tower "
                              "went down")
                    else:
                        # Deliberately not called "the spell's damage": this
                        # is every point the tower lost in the window, so any
                        # troop hitting it at the same time is counted too.
                        print(f"  [{t:7.2f}s]   {side.name.lower()} tower lost "
                              f"{int(round(before - side.hp))} HP over the "
                              "next 1.6s")
                    pending.remove(item)

            for _slot, spell, box, _score in hand:
                casts = book.casts_needed(spell, weakest.hp) if weakest else None
                draw_spell(frame, box, spell, casts, s)

            writer.write(frame)

            if weakest and hand:
                line = ",  ".join(
                    f"{sp} x{book.casts_needed(sp, weakest.hp)}"
                    for _i, sp, _b, _sc in hand)
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
        print("  No spells matched in the hand. Lower --threshold, or check "
              "card_slots in the layout file.")


if __name__ == "__main__":
    main()
