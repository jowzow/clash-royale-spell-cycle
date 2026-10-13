#!/usr/bin/env python3
"""
Regenerates data/spells.json, the crown tower damage table.

Why this script exists: Clash Royale rebalances spells constantly, so the
damage numbers have to be derivable and dated rather than typed in by hand
and forgotten. Run it again after a balance patch.

HOW THE NUMBERS ARE DERIVED
---------------------------
Supercell's own per-level damage arrays come from RoyaleAPI's cr-api-data
mirror of the game files. Two things about those arrays are not obvious:

1. They are indexed by a card's *internal* level, not its displayed level.
   The array length encodes rarity, because every rarity tops out at the
   same internal level but starts at a different one:

       Common    19 entries, starts at displayed level 1
       Rare      17 entries, starts at displayed level 3
       Epic      14 entries, starts at displayed level 6
       Legendary 11 entries, starts at displayed level 9

   So index = displayed_level - MIN_LEVEL[rarity]. Checked against the two
   Fireball figures everyone publishes: level 9 is 572 area damage and level
   11 is 689. Both land exactly where the rule says they should.

2. `crown_tower_damage_percent` is a *reduction*, stored negative. A Fireball
   at -70 is documented as doing 30% of its area damage to a crown tower.

Rule 2 does not survive contact with the footage. A level 16 Fireball has
1101 area damage, so 30% would be 330, but in videoplayback.mp4 two separate
level 16 Fireballs each take exactly 253 off an enemy princess tower:

    653.9s   left tower 2805 -> 2552
    668.0s   left tower 2552 -> 2299

253 / (1101 x 0.30) = 0.766. The game data hints at why: the Fireball record
carries `deflect_behaviour: UseSpellsTowerDamageMul`, which implies a global
spell-versus-tower multiplier living somewhere outside the per-card record.
Rather than guess at it, this script fits that single factor to the one
number the recording actually lets it measure, then applies it everywhere.

WHAT THAT MEANS FOR TRUST
-------------------------
Fireball is anchored on a measurement, seen twice. Every other spell
inherits the same global factor, which is only right if the factor really is
global. If Supercell instead changed Fireball's own percentage, the other
six are off by that 0.766. They are marked "verified": false for exactly
that reason, and the README says the same thing.

Usage:
    python tools/build_spell_table.py
    python tools/build_spell_table.py --stats cards_stats.json   # offline
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

STATS_URL = "https://royaleapi.github.io/cr-api-data/json/cards_stats.json"

MIN_LEVEL = {"Common": 1, "Rare": 3, "Epic": 6, "Legendary": 9}
LEVELS_FOR = {"Common": 19, "Rare": 17, "Epic": 14, "Legendary": 11}

# card name -> (entity name in cards_stats.json, section, rarity, elixir)
SPELLS = {
    "Rocket":         ("RocketSpell",          "projectile", "Rare",      6),
    "Fireball":       ("FireballSpell",        "projectile", "Rare",      4),
    "Lightning":      ("LighningSpell",        "projectile", "Epic",      6),
    "Arrows":         ("ArrowsSpell",          "projectile", "Common",    3),
    "The Log":        ("LogProjectileRolling", "projectile", "Legendary", 2),
    "Zap":            ("Zap",                  "spell",      "Common",    2),
    "Giant Snowball": ("SnowballSpell",        "projectile", "Common",    2),
}

# The one crown tower damage figure this project can check against footage.
MEASURED = {
    "card": "Fireball",
    "level": 16,
    "crown_damage": 253,
    "seen": ["653.9s  left tower 2805 -> 2552",
             "668.0s  left tower 2552 -> 2299"],
}

MAX_DISPLAY_LEVEL = 16


def fetch_stats(local):
    if local:
        print(f"  reading {local}")
        return json.loads(Path(local).read_text(encoding="utf-8"))
    print(f"  downloading {STATS_URL}")
    with urllib.request.urlopen(STATS_URL, timeout=120) as fh:
        return json.loads(fh.read().decode("utf-8"))


def build(stats):
    index = {}
    for section in ("spell", "projectile"):
        for entry in stats.get(section, []):
            index[(section, entry.get("name"))] = entry

    def record(card):
        entity, section, rarity, elixir = SPELLS[card]
        entry = index.get((section, entity))
        if entry is None:
            sys.exit(f"ERROR: {entity} is missing from the stats dump. The "
                     "upstream entity names may have changed.")
        damage = entry.get("damage_per_level") or []
        if len(damage) != LEVELS_FOR[rarity]:
            print(f"  WARNING: {card} has {len(damage)} levels, expected "
                  f"{LEVELS_FOR[rarity]} for {rarity}. The rarity rule may "
                  "have changed upstream.")
        pct = entry.get("crown_tower_damage_percent")
        if pct is None:
            sys.exit(f"ERROR: {card} has no crown_tower_damage_percent.")
        return rarity, elixir, damage, (100 + pct) / 100.0

    rarity, _elixir, damage, share = record(MEASURED["card"])
    i = MEASURED["level"] - MIN_LEVEL[rarity]
    if not 0 <= i < len(damage):
        sys.exit(f"ERROR: {MEASURED['card']} has no level {MEASURED['level']}.")
    factor = MEASURED["crown_damage"] / (damage[i] * share)
    print(f"  global spell-vs-tower factor fitted to the footage: {factor:.4f}"
          f"\n    ({MEASURED['card']} level {MEASURED['level']}: {damage[i]}"
          f" area x {share:.2f} x {factor:.4f} = {MEASURED['crown_damage']})")

    out = {}
    for card in SPELLS:
        rarity, elixir, damage, share = record(card)
        lo = MIN_LEVEL[rarity]
        per_level = {}
        for idx, area in enumerate(damage):
            level = idx + lo
            if level > MAX_DISPLAY_LEVEL:
                break
            per_level[str(level)] = round(area * share * factor)
        measured = card == MEASURED["card"]
        out[card] = {
            "elixir": elixir,
            "rarity": rarity,
            "min_level": lo,
            "crown_tower_damage": per_level,
            "percent_of_area_damage": round(share * factor * 100, 1),
            "verified": measured,
            "note": ("measured in videoplayback.mp4: "
                     + "; ".join(MEASURED["seen"])) if measured else
                    ("derived: the game's own per-level damage times this "
                     "card's crown tower percentage times the global factor "
                     "fitted on Fireball. Not checked against footage."),
        }
        print(f"  {card:15s} level 11 {per_level.get('11', '-'):>5}   "
              f"level 16 {per_level.get('16', '-'):>5}   "
              f"{'measured' if measured else 'derived'}")
    return out, factor


def main():
    here = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--stats", default=None,
                   help="a local copy of cards_stats.json instead of fetching")
    p.add_argument("--out", type=Path, default=here / "data" / "spells.json")
    args = p.parse_args()

    spells, factor = build(fetch_stats(args.stats))
    table = {
        "_source": STATS_URL,
        "_generated": "2026-10-12",
        "_readme": "Crown tower damage only. These are NOT area damage values.",
        "_anchor": (f"The global spell-vs-tower factor is {factor:.4f}, fitted "
                    f"so a level {MEASURED['level']} {MEASURED['card']} comes "
                    f"out at {MEASURED['crown_damage']}, which is what it does "
                    "twice in videoplayback.mp4. See tools/build_spell_table.py."),
        "_excluded": {
            "Poison": "damage over time, does not reduce to a cast count",
            "Earthquake": "damage over time, does not reduce to a cast count",
            "Tornado": "deals no meaningful crown tower damage",
        },
        "spells": spells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote {args.out}")


if __name__ == "__main__":
    main()
