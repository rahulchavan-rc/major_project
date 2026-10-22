"""
laptop_camera_test.py – Test the 4-view mango grader on your laptop webcam.

Uses the existing OpenCV capture + use_model pipeline.
A live preview window appears; press SPACE (or click the window) to
capture each side in order: Right → Left → Front → Back.
Press Q at any time to quit.

Usage:
    python laptop_camera_test.py
    python laptop_camera_test.py --camera-index 1
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import use_model  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Laptop webcam test for the MangoAI 4-view grader."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        metavar="N",
        help="Camera device index (default: 0 = built-in webcam)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MangoAI – Laptop Webcam Test")
    print("=" * 60)
    print(f"  Camera index : {args.camera_index}")
    print(f"  Capture order: Right → Left → Front → Back")
    print(f"  Controls     : SPACE or click to capture | Q to quit")
    print("=" * 60 + "\n")

    try:
        use_model.capture_photos(camera_index=args.camera_index)
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        print("Make sure your webcam is connected and not in use by another app.")
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
