#!/usr/bin/env python3
"""
Checks the spell damage table and the cast arithmetic.

This is the part of the project that can be verified without a video, so it
is worth pinning down: if these numbers are wrong, every count drawn on
screen is wrong too.

The one number here that is measured rather than derived is Fireball at
level 16. Two casts in videoplayback.mp4 each take exactly 253 off a crown
tower, and the rest of the table is scaled from that; see
tools/build_spell_table.py. tools/test_health.py checks the other half, the
health reading, against the recording.

Usage:
    python tools/test_math.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spellcycle import SpellBook  # noqa: E402

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
lvl16 = SpellBook(DATA, 16)

print("\nThe measured anchor: a level 16 Fireball on a crown tower")
# Seen twice in videoplayback.mp4: 2805 -> 2552 and 2552 -> 2299.
check("Fireball at 16", lvl16.damage["Fireball"], 253)

print("\nThe rest of the table scales with the game's own damage arrays")
check("Fireball at 11", book.damage["Fireball"], 158)
check("Rocket at 16 > Fireball at 16",
      lvl16.damage["Rocket"] > lvl16.damage["Fireball"], True)
check("Zap at 16 < Fireball at 16",
      lvl16.damage["Zap"] < lvl16.damage["Fireball"], True)

print("\nCasts to delete the tower the demo is aimed at (3325 HP)")
hp = 3325
check("Fireball", book.casts_needed("Fireball", hp), 22)      # 3325/158
check("Fireball at 16", lvl16.casts_needed("Fireball", hp), 14)  # 3325/253
check("Rocket at 16", lvl16.casts_needed("Rocket", hp), 8)     # 3325/454

print("\nPartial health rounds up, never down")
dmg = lvl16.damage["Fireball"]
check("1 HP left needs 1 cast", lvl16.casts_needed("Fireball", 1), 1)
check(f"exactly {dmg} needs 1", lvl16.casts_needed("Fireball", dmg), 1)
check(f"{dmg + 1} needs 2", lvl16.casts_needed("Fireball", dmg + 1), 2)
check(f"{2 * dmg} needs 2", lvl16.casts_needed("Fireball", 2 * dmg), 2)

print("\nEdge cases return None rather than crashing")
check("dead tower", lvl16.casts_needed("Fireball", 0), None)
check("negative HP", lvl16.casts_needed("Fireball", -50), None)
check("health not readable", lvl16.casts_needed("Fireball", None), None)
check("unknown spell", lvl16.casts_needed("Mega Knight", 3325), None)

print("\nHigher level spells hit harder")
check("Rocket 16 > 11", lvl16.damage["Rocket"] > book.damage["Rocket"], True)
check("fewer casts at 16",
      lvl16.casts_needed("Rocket", hp) <= book.casts_needed("Rocket", hp), True)
low = SpellBook(DATA, 9)
check("Rocket 9 < 11", low.damage["Rocket"] < book.damage["Rocket"], True)

print("\nEvery spell has a level 16 entry, since that is what ladder reaches")
for name in book.spells:
    check(f"{name} at 16", str(16) in book.spells[name]["crown_tower_damage"], True)

print(f"\n  {passed} passed, {failed} failed\n")
sys.exit(1 if failed else 0)
