#!/usr/bin/env python3
"""
Fine-tunes YOLOv8n to detect spells in the arena.

Two classes:
    spell_in_flight  the projectile crossing the arena
    spell_impact     the blast where it lands

Why a model at all, when the rest of the program is classical OpenCV: the
hand and the health bars sit at fixed screen positions, so template matching
and a fixed rectangle are exactly right for them. A spell in flight has no
fixed position, no fixed scale, and flies over whatever happens to be on the
board. That is the case a detector is for.

Starting from COCO-pretrained yolov8n rather than from scratch: a few hundred
frames is nowhere near enough to learn edges and blobs from nothing, but it
is plenty to re-point a pretrained backbone at one new object.

Usage:
    python tools/train_model.py
    python tools/train_model.py --epochs 60
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "dataset" / "data.yaml")
    # The frames are 298x640. Padding those to a 640 square would spend more
    # than half the compute on grey bars, so training is rectangular: the long
    # side is imgsz and the short side follows the real aspect ratio.
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--base", default="yolov8n.pt")
    args = p.parse_args()

    if not args.data.exists():
        sys.exit(f"ERROR: {args.data} missing. Run:\n"
                 "    python tools/build_dataset.py videoplayback.mp4")

    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        project=str(ROOT / "runs"),
        name="spell",
        exist_ok=True,
        seed=7,
        pretrained=True,
        patience=12,
        # The projectile is small and the frames are already game renders, so
        # the heavy photographic augmentations hurt more than they help.
        hsv_h=0.010, hsv_s=0.3, hsv_v=0.3,
        degrees=0.0, shear=0.0, perspective=0.0,
        # Mosaic stitches four frames into one square, which reintroduces the
        # padding rectangular training exists to avoid, so it stays off.
        mosaic=0.0, mixup=0.0, translate=0.08, scale=0.3,
        fliplr=0.5, flipud=0.0,
        rect=True,
        workers=8,
        verbose=True,
    )

    metrics = model.val(data=str(args.data), device="cpu", imgsz=args.imgsz)
    print("\n  === held-out results (validation events never seen in training) ===")
    print(f"  mAP50      {metrics.box.map50:.3f}")
    print(f"  mAP50-95   {metrics.box.map:.3f}")
    for i, name in enumerate(metrics.names.values()):
        print(f"  {name:16s} AP50 {metrics.box.ap50[i]:.3f}"
              f"   P {metrics.box.p[i]:.3f}   R {metrics.box.r[i]:.3f}")

    best = ROOT / "runs" / "spell" / "weights" / "best.pt"
    if best.exists():
        dest = ROOT / "models" / "spell_yolov8n.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(best.read_bytes())
        print(f"\n  Copied best weights to {dest}")


if __name__ == "__main__":
    main()
