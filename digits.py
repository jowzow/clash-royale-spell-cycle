#!/usr/bin/env python3
"""
Reads the health number Clash Royale prints above a crown tower's bar.

The number is about seven pixels tall in a phone capture, which is far too
small for anything that expects to recognise text: a first attempt at this
read a clear 4858 as 3038, and the project spent a while measuring the
health bar instead. But the game draws the number in one fixed font at one
fixed size, which turns the problem from "recognise text" into "match ten
known glyphs", and that is tractable even at this size.

How a strip is read:

  1. Normalise. The number is white on whatever the arena happens to be, so
     brightness is rescaled against the strip's own 20th and 99th percentile
     and anything strongly coloured is dropped. That kills the gold level
     badge sitting just to the left.
  2. Line up vertically. The HUD shifts a pixel between matches and one row
     of slip turns a 5 into an 8, so the strip is slid until the ink's row
     centroid lands where the templates expect it.
  3. Place glyphs left to right. Each digit has its own advance width (a 1 is
     narrower than an 8), so the search walks the string one glyph at a time,
     keeping the best scoring parse at each pixel position.
  4. Check the whole string. The winning parses are re-drawn from the
     templates and compared against the strip as a picture. A parse that
     leaves ink unexplained, or paints ink where there is none, loses here
     even when every single glyph scored well.

tools/build_digits.py cuts the templates out of a recording and writes
data/digits.json. Both the builder and the reader use this module, so the
two can never drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

DIGITS = "0123456789"

BAND = (110, 126)      # rows of the frame the number can appear in
SPAN = 32              # columns of the strip, from the tower's anchor
TH, TW = 9, 7          # template size in source pixels
Z = 4                  # the strip is matched at 4x, for sub-pixel placement
THU, TWU = TH * Z, TW * Z
HALF = (TW - 1) / 2


# --------------------------------------------------------------------------
# Turning a strip of frame into something matchable
# --------------------------------------------------------------------------


def prep(bgr, dy: int = 0):
    """-> (soft ink 0..1, binary mask), both cropped to TH rows. (None, None)
    when the strip holds no text at all, which is what a destroyed tower and
    a between-matches screen both look like."""
    if bgr is None or bgr.shape[0] < TH:
        return None, None
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    lo, hi = np.percentile(g, 20), np.percentile(g, 99)
    soft = np.clip((g - lo) / max(hi - lo, 1.0), 0, 1)
    soft[sat > 110] = 0.0
    cut = max(165.0, lo + 0.62 * (hi - lo))
    mask = ((g >= cut) & (sat <= 110)).astype(np.float32)

    prof = mask.sum(axis=1)
    if prof.sum() < 6:
        return None, None
    centre = float((prof * np.arange(len(prof))).sum() / prof.sum())
    top = int(round(centre)) - TH // 2 + dy
    top = max(0, min(top, mask.shape[0] - TH))
    return soft[top:top + TH], mask[top:top + TH]


def extent(mask):
    """Columns the number occupies, as (first, last+1), or None."""
    col = mask.sum(axis=0)
    on = col >= 1.0
    # A digit is at least two columns wide, so a lone lit column is
    # compression fringe or the corner of the level badge. Left in, it drags
    # the start of the number sideways and every glyph after it lands wrong.
    for x in range(len(on)):
        if on[x] and not (x and on[x - 1]) and not (x + 1 < len(on) and on[x + 1]):
            on[x] = False
    idx = np.flatnonzero(on)
    if idx.size == 0:
        return None
    lo = hi = int(idx[0])
    gap = 0
    for x in range(lo, len(col)):
        if on[x]:
            hi, gap = x + 1, 0
        else:
            gap += 1
            if gap >= 3:         # two blank columns still happen between digits
                break
    while lo < hi and col[lo] < 2:
        lo += 1
    while hi > lo and col[hi - 1] < 2:
        hi -= 1
    return (lo, hi) if hi - lo >= 2 else None


def upscale(soft):
    return cv2.resize(soft, None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC)


def ncc(a, b) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-9 else 0.0


# --------------------------------------------------------------------------
# Splitting a strip whose value is already known, for template building
# --------------------------------------------------------------------------


def split(mask, lo, hi, n):
    """Cut the number into n digit column ranges, cutting where the ink is
    thinnest. Only used when building templates, where n is known."""
    col = mask.sum(axis=0)
    if n == 1:
        return [(lo, hi)]
    memo = {}

    def best(a, k):
        if k == 1:
            return (0.0, [(a, hi)]) if 2 <= hi - a <= 7 else (1e9, None)
        if (a, k) in memo:
            return memo[(a, k)]
        out = (1e9, None)
        for w in range(2, 8):
            b = a + w
            if b >= hi:
                break
            cost, path = best(b, k - 1)
            if path is not None and col[b] + cost < out[0]:
                out = (col[b] + cost, [(a, b)] + path)
        memo[(a, k)] = out
        return out

    return best(lo, n)[1]


def centroid(mask, a, b) -> float:
    c = mask[:, a:b].sum(axis=0)
    if c.sum() <= 0:
        return (a + b - 1) / 2
    return a + float((c * np.arange(b - a)).sum() / c.sum())


# --------------------------------------------------------------------------
# The reader
# --------------------------------------------------------------------------


class DigitReader:
    """Reads a health number out of a strip of frame."""

    def __init__(self, templates: dict, advances: dict, min_score: float = 0.86):
        self.t = {d: np.asarray(v, dtype=np.float32) / 255.0
                  for d, v in templates.items()}
        self.adv = {d: float(v) for d, v in advances.items()}
        self.min_score = min_score

    @classmethod
    def load(cls, path: Path, min_score: float = 0.86) -> "DigitReader":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(blob["templates"], blob["advances"], min_score)

    def save(self, path: Path, note: str = "") -> None:
        Path(path).write_text(json.dumps({
            "_note": note,
            "_shape": [THU, TWU],
            "templates": {d: np.round(v * 255).astype(int).tolist()
                          for d, v in self.t.items()},
            "advances": {d: round(v, 3) for d, v in self.adv.items()},
        }), encoding="utf-8")

    def read(self, bgr):
        """-> (value as int, score) or (None, score). None means 'could not
        read it', never 'the tower is at zero'.

        The score threshold is set where it separates best on this recording:
        of the readings the decoder is willing to return, a correct one
        scores 0.93 at the median and a wrong one 0.84, so cutting at 0.86
        throws away three quarters of the wrong readings for 15% of the
        right ones. Thrown-away frames cost nothing, because the caller is
        voting over half a second and holds its last value meanwhile."""
        best = (None, 0.0)
        for dy in (0, -1, 1):
            v, s = self._read_at(bgr, dy)
            if v is not None and s > best[1]:
                best = (v, s)
        return best if best[1] >= self.min_score else (None, best[1])

    def _read_at(self, bgr, dy):
        soft, mask = prep(bgr, dy)
        if soft is None:
            return None, 0.0
        found = extent(mask)
        if found is None:
            return None, 0.0
        lo, hi = found
        if not 2 <= hi - lo <= 24:        # longer than four digits is not text
            return None, 0.0

        up = upscale(soft)
        score = {d: cv2.matchTemplate(up, t, cv2.TM_CCOEFF_NORMED)[0]
                 for d, t in self.t.items()}
        last = len(next(iter(score.values()))) - 1

        def at(d, c):
            i = int(round((c - HALF) * Z))
            return float(score[d][i]) if 0 <= i <= last else None

        # Walk the string left to right. Parses that reach the same pixel
        # column are interchangeable from there on, so only the best one at
        # each column is carried forward.
        parses, frontier = [], {}

        def carry(store, c, total, path):
            key = round(c * 4)
            if key not in store or store[key][0] < total:
                store[key] = (total, path, c)

        for c0 in np.arange(lo + 0.75, lo + 3.01, 0.25):
            for d, s in ((d, at(d, c0)) for d in self.t):
                if s is None:
                    continue
                parses.append((s, [(d, c0)]))
                carry(frontier, c0 + self.adv[d], s, [(d, c0)])
        for _ in range(3):
            nxt = {}
            for total, path, c in frontier.values():
                for d in self.t:
                    s = at(d, c)
                    if s is None:
                        continue
                    step = path + [(d, c)]
                    parses.append(((total + s) / len(step), step))
                    carry(nxt, c + self.adv[d], total + s, step)
            frontier = nxt
            if not frontier:
                break
        if not parses:
            return None, 0.0

        # Re-draw the best few and judge them as whole pictures. This is what
        # catches a parse that reads four good glyphs but leaves a fifth
        # digit's ink sitting unexplained at the end.
        parses.sort(key=lambda p: -p[0])
        best = (None, 0.0)
        seen = set()
        for mean, path in parses[:12]:
            text = "".join(d for d, _ in path)
            if text in seen or (len(text) > 1 and text[0] == "0"):
                continue
            seen.add(text)
            canvas = np.zeros_like(up)
            for d, c in path:
                left = int(round((c - HALF) * Z))
                a, b = max(0, left), min(up.shape[1], left + TWU)
                if b > a:
                    canvas[:, a:b] = np.maximum(canvas[:, a:b],
                                                self.t[d][:, a - left:b - left])
            total = 0.35 * mean + 0.65 * ncc(canvas, up)
            if total > best[1]:
                best = (int(text), total)
        return best
