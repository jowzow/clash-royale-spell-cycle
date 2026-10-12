#!/usr/bin/env python3
"""
Checks the spell damage table and the cast arithmetic.

This is the part of the project that can be verified without a video, so it
is worth pinning down: if these numbers are wrong, every count drawn on
screen is wrong too.

Usage:
    python tools/test_math.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spellcycle import PRINCESS_HP, SpellBook  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "spells.json"

passed = failed = 0


def check(label, got, want):
    global passed, failed
    if got == want:
        passed += 1
        print(f"  ok    {label}: {got}")
    else:
        failed += 1
        print(f"  FAIL  {label}: got {got}, want {want}")


book = SpellBook(DATA, 11)

print("\nCrown tower damage at level 11 (tournament standard)")
# Published post-patch values, the two anchors the whole table is built on.
check("Fireball", book.damage["Fireball"], 159)
check("Rocket", book.damage["Rocket"], 343)
check("Lightning", book.damage["Lightning"], 295)

print("\nCasts to delete a level 11 princess tower (3052 HP)")
hp = PRINCESS_HP[11]
check("princess HP", hp, 3052)
check("Rocket", book.casts_needed("Rocket", hp), 9)       # 3052/343 = 8.90
check("Fireball", book.casts_needed("Fireball", hp), 20)  # 3052/159 = 19.19
check("Lightning", book.casts_needed("Lightning", hp), 11)  # 3052/295 = 10.35

print("\nCasts against a level 15 tower (4424 HP)")
check("level 15 HP", PRINCESS_HP[15], 4424)
check("Rocket", book.casts_needed("Rocket", PRINCESS_HP[15]), 13)

print("\nPartial health rounds up, never down")
check("1 HP left needs 1 cast", book.casts_needed("Rocket", 1), 1)
check("exactly 343 needs 1", book.casts_needed("Rocket", 343), 1)
check("344 needs 2", book.casts_needed("Rocket", 344), 2)
check("686 needs 2", book.casts_needed("Rocket", 686), 2)

print("\nEdge cases return None rather than crashing")
check("dead tower", book.casts_needed("Rocket", 0), None)
check("negative HP", book.casts_needed("Rocket", -50), None)
check("unknown spell", book.casts_needed("Mega Knight", 3052), None)

print("\nHigher level spells hit harder")
high = SpellBook(DATA, 14)
check("Rocket 14 > 11", high.damage["Rocket"] > book.damage["Rocket"], True)
check("fewer casts at 14",
      high.casts_needed("Rocket", hp) <= book.casts_needed("Rocket", hp), True)
low = SpellBook(DATA, 9)
check("Rocket 9 < 11", low.damage["Rocket"] < book.damage["Rocket"], True)

print(f"\n  {passed} passed, {failed} failed\n")
sys.exit(1 if failed else 0)
