#!/usr/bin/env python3
"""
Regenerates data/spells.json, the crown tower damage table.

Why this script exists: Clash Royale rebalances spells constantly, so the
damage numbers must be derivable and dated rather than typed in by hand and
forgotten. Run it again after a balance patch and edit CURRENT_L11 below.

HOW THE NUMBERS ARE DERIVED
---------------------------
Supercell's own per-level damage arrays come from RoyaleAPI's cr-api-data
mirror of the game files. Two things about those arrays are not obvious:

1. They are indexed by a card's *internal* level, not its displayed level.
   The array length encodes rarity, because every rarity tops out at the same
   internal level but starts at a different one:

       Common    19 entries, starts at displayed level 1
       Rare      17 entries, starts at displayed level 3
       Epic      14 entries, starts at displayed level 6
       Legendary 11 entries, starts at displayed level 9

   So index = displayed_level - MIN_LEVEL[rarity].

2. `crown_tower_damage_percent` is a *reduction*, stored negative. A Fireball
   at -70 deals 30% of its area damage to a Crown Tower.

Both rules were verified against two independently published values:
   Fireball level 11 -> 689 area damage, 207 crown tower damage
   Rocket   level 11 -> 1484 area damage, 371 crown tower damage
Both reproduce exactly, which is why the indexing is trusted.

The mirror itself was last updated in 2023, so its crown tower numbers are
pre-2026. The June 2026 balance patch cut the crown tower damage of twelve
spells, and September 2026 cut Fireball again. CURRENT_L11 below holds the
post-patch level 11 values; everything else is scaled from them using the
game's own per-level damage ratios, which preserves exact level scaling
without inventing anything.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

STATS_URL = "https://royaleapi.github.io/cr-api-data/json/cards_stats.json"

MIN_LEVEL = {"Common": 1, "Rare": 3, "Epic": 6, "Legendary": 9}

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

# Crown tower damage at level 11 after the 2026 balance patches.
#   verified=True  -> the post-patch number itself is published
#   verified=False -> published pre-patch value with the published percentage
#                     cut applied, so it can be off by a point of rounding
CURRENT_L11 = {
    "Rocket":         (343, True,  "June 2026: -8% crown tower damage"),
    "Fireball":       (159, True,  "June 2026 -17%, then September 2026 again"),
    "Lightning":      (295, False, "June 2026: -7% applied to 317"),
    "Arrows":         (30,  False, "June 2026: -19% applied to 37"),
    "The Log":        (49,  False, "June 2026: -15% applied to 58"),
    "Zap":            (48,  False, "June 2026: -17% applied to 58"),
    "Giant Snowball": (48,  False, "June 2026: -17% applied to 58"),
}

MAX_DISPLAY_LEVEL = 15


def fetch_stats() -> dict:
    print(f"  downloading {STATS_URL}")
    with urllib.request.urlopen(STATS_URL, timeout=120) as fh:
        return json.loads(fh.read().decode("utf-8"))


def build() -> dict:
    stats = fetch_stats()
    index = {}
    for section in ("spell", "projectile"):
        for entry in stats.get(section, []):
            index[(section, entry.get("name"))] = entry

    out = {}
    for card, (entity, section, rarity, elixir) in SPELLS.items():
        entry = index.get((section, entity))
        if entry is None:
            sys.exit(f"ERROR: {entity} missing from the stats dump.")

        damage = entry.get("damage_per_level") or []
        expected = {"Common": 19, "Rare": 17, "Epic": 14, "Legendary": 11}
        if len(damage) != expected[rarity]:
            print(f"  WARNING: {card} has {len(damage)} levels, expected "
                  f"{expected[rarity]} for {rarity}. Rarity rule may have "
                  f"changed upstream.")

        lo = MIN_LEVEL[rarity]
        i11 = 11 - lo
        if not 0 <= i11 < len(damage):
            sys.exit(f"ERROR: {card} has no level 11 entry.")

        ct11, verified, note = CURRENT_L11[card]
        # Anchor on the verified level 11 value and scale by the game's own
        # damage ratios, so every other level keeps exact in-game scaling.
        per_level = {}
        for i, dmg in enumerate(damage):
            level = i + lo
            if level > MAX_DISPLAY_LEVEL:
                break
            per_level[str(level)] = round(ct11 * dmg / damage[i11])

        out[card] = {
            "elixir": elixir,
            "rarity": rarity,
            "min_level": lo,
            "crown_tower_damage": per_level,
            "area_damage_level_11": damage[i11],
            "verified": verified,
            "note": note,
        }
        print(f"  {card:15s} L11 crown tower damage = {per_level['11']:4d}"
              f"   {'verified' if verified else 'derived'}")
    return out


def main():
    table = {
        "_source": STATS_URL,
        "_generated": "2026-10-12",
        "_balance": "Reflects the June 2026 and September 2026 spell patches.",
        "_readme": "Crown tower damage only. These are NOT area damage values.",
        "_excluded": {
            "Poison": "damage over time, does not reduce to a cast count",
            "Earthquake": "damage over time, does not reduce to a cast count",
            "Tornado": "deals no meaningful crown tower damage",
        },
        "spells": build(),
    }
    out = Path(__file__).resolve().parent.parent / "data" / "spells.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"\n  Wrote {out}")


if __name__ == "__main__":
    main()
