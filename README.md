# Spell Cycle Counter

**How many casts does it take to delete that tower?**

Spell cycling is the Clash Royale plan where you ignore the board and chip a
tower down with spells alone. It works because spell damage to a crown tower
is guaranteed: no defence, no blocking, no trading. So the number of casts is
not a guess, it is arithmetic.

This program does that arithmetic for you, live, off a recorded match. It
reads the enemy princess towers' health off their health bars, finds which
spells are sitting in your hand, and prints a number on each one: how many
casts of that spell finish off the weaker tower.

![The counter running on a real match](output/demo.gif)

The Fireball in hand is boxed and labelled. The count ticks down as the tower
takes damage, because a level 11 Fireball does 159 to a crown tower and the
tower has less health every time it gets hit.

Full annotated clip: [`output/videoplayback_spellcycle.mp4`](output/videoplayback_spellcycle.mp4)
(85 seconds, 2548 frames). Single frame: [`output/demo_frame.png`](output/demo_frame.png).

---

## What it detects

Two things, both in the game's heads-up display rather than in the arena:

1. **The enemy princess towers' health**, from the red fill of their health bars.
2. **Which of your four hand slots hold a damaging spell**, from the card art.

Non-spell cards are deliberately ignored. A Goblin Gang has no guaranteed
tower damage, so no count can be put on it.

## Approach, and why

**No neural network.** Both targets live in a fixed, rendered user interface,
not in a photograph. A YOLO-style detector would be a worse tool here: it
would need training data, and it would be less accurate than reading pixels
whose exact position and colour the game guarantees. Two classical methods do
the job with OpenCV alone, which is why `requirements.txt` is two lines.

**Hand: template matching against the official card art.** The four slots sit
at fixed screen positions, so each slot is matched against all seven spell
images rather than sliding templates over the whole frame. That is faster and
it structurally cannot put a card in the wrong slot. Card art is downloaded
from RoyaleAPI's asset mirror into `cards/`.

The separation is clean. In the demo frames:

| Case | Match score |
|---|---|
| Fireball really in hand | 0.86 to 0.92 |
| Best false positive (greyed-out post-match card) | 0.72 |
| Everything else | below 0.56 |

Hence the default `--threshold 0.78`.

**Health: measuring the bar, not reading the number.** The game does print the
health as a number, and that was the first thing I tried. It does not survive
this recording. At 298x640 the digits are six pixels tall; I built digit
templates from values I read by eye and tested them on held-out frames, and
`4858` came back as `3038`. That is a resolution limit, not a tuning problem,
so I measured the bar instead. Fitted against four hand-read health values,
the bar lands within 0.3%.

## How the code processes the video

`spellcycle.py` runs one loop per frame:

1. **Read each enemy bar.** Convert the bar rectangle to HSV, mask the two red
   hue ranges, and measure how far the filled run reaches from the left edge.
2. **Smooth it.** Covered in detail below, because this is where the real work is.
3. **Pick the target.** The weakest tower still standing. A destroyed tower is
   not a zero-health target, it is removed from the running.
4. **Read the hand** every fifth frame, since a hand only changes when a card
   is played.
5. **Do the arithmetic.** `ceil(health / crown tower damage)`, which always
   rounds up, because a tower on 1 health still needs a whole cast.
6. **Draw** a box on each spell with its count, and nothing else. The health
   figures stay off the picture on purpose: they are an estimate good to about
   a bar pixel, which is accurate enough to pick a cast count but not
   accurate enough to put on screen as if it were the game's own number.

### Keeping the health reading honest

A health bar gets walked over by troops, covered by spell animations, and
hidden under the card you are dragging. Three rules handle all of it, and
together they make the estimate monotonic:

- **A hidden bar is unknown, not empty.** If no red is found the frame is
  discarded. Returning zero here was a real bug: the first spell animation
  over the bar latched the tower to dead.
- **The largest recent reading wins.** Anything covering a bar can only make
  the red run look *shorter* than it is, never longer. So across half a second
  the maximum is the honest value. Replacing a median with this one line
  removed every remaining glitch.
- **Towers never heal.** The estimate can only ever fall.

When the bar is unreadable the last good value simply stands. Four seconds of
continuous absence means the tower is gone, since nothing covers a bar that
long but destruction.

### Where the damage numbers come from

This was the fiddliest part to get right, and `tools/build_spell_table.py`
documents it in full. Two traps:

- **Crown tower damage is not area damage.** Spells deal heavily reduced
  damage to towers. A level 11 Fireball does 689 to troops but 159 to a tower.
  Using the wrong one would overstate every count by four times.
- **Every public stat dump is stale.** June 2026 cut the crown tower damage of
  twelve spells at once, and September 2026 cut Fireball again. The game-file
  mirror still carries pre-patch values.

So the table is anchored on the two post-patch values that are published
outright, and every other level is scaled using the game's own per-level
damage ratios. The indexing rule those arrays use is not documented anywhere
I could find, so I derived it and then verified it: array length encodes
rarity, and index equals level minus the rarity's starting level (Common 1,
Rare 3, Epic 6, Legendary 9). Checking it reproduces both published pre-patch
values exactly, 207 for Fireball and 371 for Rocket.

Crown tower damage at level 11, tournament standard:

| Spell | Damage | Casts for a 3052 health tower |
|---|---|---|
| Rocket | 343 | 9 |
| Lightning | 295 | 11 |
| Fireball | 159 | 20 |
| The Log | 49 | 63 |
| Zap | 48 | 64 |
| Giant Snowball | 48 | 64 |
| Arrows | 30 | 102 |

Poison and Earthquake are deliberately excluded. They deal damage over time,
so "how many casts" has no honest answer for them.

## What the custom behaviour does

Detecting things is the easy half. The custom logic is everything that turns
two detections into one actionable number:

- **Picks a target.** The weakest princess tower still standing. A destroyed
  tower is removed from the running rather than treated as a zero-health
  target, which would otherwise make every count read 0.
- **Converts health into casts.** `ceil(health / crown tower damage)`, always
  rounding up, because a tower on 1 health still costs a whole cast.
- **Scales to the spell's level.** `--spell-level` re-reads the whole damage
  table, so a level 14 Rocket correctly needs fewer casts than a level 11 one.
- **Labels the right card.** Each count is drawn on the specific hand slot
  holding that spell, and only on spells, since nothing else does guaranteed
  tower damage.
- **Survives a messy video.** The three rules under "Keeping the health
  reading honest" above are the real work: a bar hidden by a troop or a spell
  animation holds its last value instead of collapsing the count.
- **Logs every change.** A line is printed whenever the count changes, so the
  run is auditable without scrubbing the video:

```
[ 751.22s] Fireball x21
[ 756.52s] Fireball x20
[ 759.66s] Fireball x19
```

## When nothing is detected

Nothing detected is a normal state here, not an error, and every case is
handled without a traceback:

| Situation | What happens |
|---|---|
| No spell in your hand | nothing is drawn, the video keeps going |
| A bar is covered by a troop or spell | the frame is discarded, last good value holds |
| A tower is destroyed | dropped from targeting after its bar is gone 4 seconds |
| Both towers gone | no target, counts stop being drawn |
| Greyed-out post-match cards | rejected by the match threshold |
| Video file missing or undecodable | one-line error, clean exit |
| `--start` past the end of the video | one-line error, previous output left intact |
| `--spell-level` out of range | one-line error naming the valid range |

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Built and tested on Python 3.12.1, Windows 11.

## Run

```bash
# the demo segment. 4858 is the max tower health in this recording, read off
# a full enemy bar; without it the program assumes a level 11 tower and warns.
python spellcycle.py videoplayback.mp4 --start 700 --end 785 --tower-hp 4858

# your own spell levels
python spellcycle.py match.mp4 --spell-level 14

# check the maths without touching a video
python tools/test_math.py

# rebuild the damage table after a balance patch
python tools/build_spell_table.py
```

Output lands in `output/` as an annotated MP4.

| Flag | Meaning |
|---|---|
| `--spell-level` | your spell level, default 11 |
| `--tower-hp` | enemy tower max health, default 3052 (a level 11 tower) |
| `--start` / `--end` | seconds, to cut one match out of a long recording |
| `--threshold` | card match score, default 0.78 |
| `--every` | re-read the hand every N frames, default 5 |

### Using a different recording

The on-screen regions live in `data/layout.json` as fractions of the frame, so
nothing is hard-coded to one resolution. For a new capture:

```bash
python tools/inspect_frame.py yourvideo.mp4 --grid
```

Read the percentages off the grid it draws, then edit the four card slots and
two bar rectangles to match.

## Sample output

Everything in `output/` is committed, produced by the demo command above:

| File | What it is |
|---|---|
| `demo.gif` | the 10 second excerpt shown at the top |
| `videoplayback_spellcycle.mp4` | the full 85 second annotated clip |
| `demo_frame.png` | a single annotated frame |

The source recording itself is not committed. It is a 68 MB download and the
repository does not need it to be read, only to be re-run.

## Verification

The health estimate drives every count, so it was checked against the health
the game itself prints on the demo segment:

| Tower | Estimated | Game shows | Error |
|---|---|---|---|
| Left | 2969 | 3028 | 59 |
| Right | 4588 | 4572 | 16 |

Both are inside one pixel of bar, which is about 131 health on these towers.
That is why the estimate is used for the arithmetic but never drawn on the
video as a health readout.
`tools/test_math.py` covers the damage table and the cast arithmetic with 19
assertions, including that partial health always rounds up and that dead
towers and unknown spells return nothing instead of crashing.

## Limitations

- **Health is read to about plus or minus 131 points**, one bar pixel on a
  37 pixel bar. That can shift a count by one near a boundary. A higher
  resolution recording would fix this outright by making the printed health
  number readable, and that is the first thing I would add.
- **Maximum tower health is assumed, not detected.** The default is a level 11
  tower at 3052. The demo video has level 16 towers, so it is run with
  `--tower-hp 4858`, a value I read off a full bar in that match.
- **Seven spells are supported.** Adding one is a line in
  `tools/build_spell_table.py` plus its card art.
- **The king tower is not tracked.** It prints no health number and its bar is
  partly behind the king sprite. Spell cycling targets a princess tower anyway.
- **Royal Chef can heal a tower**, which breaks the never-heals assumption. It
  is rare enough that the robustness it buys everywhere else is worth it.
- **One layout per recording.** A different aspect ratio needs the regions
  re-measured.

## What I would add next

- Read the printed health number on a higher resolution capture and use the
  bar only as a fallback. That removes the `--tower-hp` assumption entirely.
- Detect the tower level badge to set maximum health automatically.
- Count elixir, so the readout could say whether the cycle is actually
  affordable before the match ends.
- Handle Poison and Earthquake as "chip per cast over time" rather than
  excluding them.
