"""
rpi_camera.py – Raspberry Pi mango grader with OLED display.

Auto-detects camera:
  1. Raspberry Pi Camera Module via picamera2 (libcamera)
  2. USB webcam via OpenCV (cv2)

Captures 4 views (Right → Left → Front → Back) using a physical button
(GPIO 17, active-low) or keyboard Enter, then predicts the mango grade
using the trained model and shows the final result on a 1.3-inch
SSD1306 OLED display (128×64, I2C).

Hardware wiring (Raspberry Pi GPIO):
  OLED (SSD1306, I2C):
    GND  → Pin 6  (GND)
    VCC  → Pin 1  (3.3 V)
    SCL  → Pin 5  (GPIO 3 / SCL1)
    SDA  → Pin 3  (GPIO 2 / SDA1)
  Button (optional):
    One leg → Pin 11 (GPIO 17)
    Other   → Pin 9  (GND)

Usage:
    python rpi_camera.py
    python rpi_camera.py --camera-index 1
    python rpi_camera.py --no-display
    python rpi_camera.py --no-button
"""

from __future__ import annotations

import argparse
import select
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import use_model  # noqa: E402

CAMERA_SIDES  = ("right", "left", "front", "back")
OLED_WIDTH    = 128
OLED_HEIGHT   = 64
BUTTON_PIN    = 17
CAPTURE_DELAY = 0.25


# ── OLED helpers ──────────────────────────────────────────────────────────────

def _load_oled():
    try:
        from luma.core.interface.serial import i2c as luma_i2c
        from luma.oled.device import ssd1306
        serial = luma_i2c(port=1, address=0x3C)
        device = ssd1306(serial, width=OLED_WIDTH, height=OLED_HEIGHT)
        print("[OLED] SSD1306 found via luma.oled at 0x3C.")
        return device, "luma"
    except Exception:
        pass

    try:
        import Adafruit_SSD1306
        disp = Adafruit_SSD1306.SSD1306_128_64(rst=None)
        disp.begin()
        disp.clear()
        disp.display()
        print("[OLED] SSD1306 found via Adafruit_SSD1306.")
        return disp, "adafruit"
    except Exception:
        pass

    print("[OLED] No OLED display found. Install luma.oled or Adafruit_SSD1306.")
    return None, None


def _oled_render(device, driver: str, lines: list[str]) -> None:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    image = PILImage.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)
    draw  = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    line_h = 14
    for i, line in enumerate(lines[:4]):
        draw.text((2, i * line_h + 2), line, fill=255, font=font)
    if driver == "luma":
        device.display(image)
    elif driver == "adafruit":
        device.image(image)
        device.display()


def _oled_clear(device, driver: str) -> None:
    if device is None:
        return
    if driver == "luma":
        from luma.core.render import canvas as lc
        with lc(device) as draw:
            draw.rectangle(device.bounding_box, outline=0, fill=0)
    elif driver == "adafruit":
        device.clear()
        device.display()


def _oled_status(device, driver: str, l1: str, l2: str = "") -> None:
    if device is None:
        return
    _oled_render(device, driver, ["  MangoAI", "-" * 16, f"  {l1}", f"  {l2}"])


def _oled_result(device, driver: str, result: dict) -> None:
    if device is None:
        return
    status = result.get("status", "")
    grade  = result.get("predicted_grade")

    if status == "NON_MANGO" or not grade:
        # Validity model confirmed it's NOT a mango
        lines = ["  MangoAI", "-" * 16, "  INVALID", "  Not a mango"]
    else:
        # Grade predicted (validity model may or may not have been available)
        conf  = result.get("grade_confidence", 0) * 100
        votes = result.get("winning_votes", "?")
        note  = "No validity chk" if status == "VALIDITY_UNAVAILABLE" else "Verified"
        lines = [
            "  MangoAI Result",
            "-" * 16,
            f"  {grade.title()}",
            f"  {conf:.1f}%  {votes}/4 votes",
        ]
    _oled_render(device, driver, lines)


# ── Camera helpers ────────────────────────────────────────────────────────────

class _PiCamera:
    def __init__(self):
        from picamera2 import Picamera2
        self._cam = Picamera2()
        cfg = self._cam.create_still_configuration(
            main={"size": (1920, 1080), "format": "RGB888"},
        )
        self._cam.configure(cfg)
        self._cam.start()
        time.sleep(1.5)
        print("[Camera] Raspberry Pi Camera Module (picamera2) ready.")

    def capture_frame(self):
        return self._cam.capture_array("main")

    def release(self):
        self._cam.stop()
        self._cam.close()


class _UsbCamera:
    def __init__(self, index: int):
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"Cannot open USB camera index {index}.")
        for _ in range(5):
            self._cap.read()
        print(f"[Camera] USB webcam at index {index} (OpenCV) ready.")

    def capture_frame(self):
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Failed to read USB camera frame.")
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def release(self):
        self._cap.release()


def _open_camera(camera_index: int):
    try:
        return _PiCamera()
    except Exception as exc:
        print(f"[Camera] picamera2 unavailable: {exc}")
    try:
        return _UsbCamera(camera_index)
    except Exception as exc:
        raise RuntimeError(
            "Could not open any camera.\n"
            "  Pi Camera : sudo apt install python3-picamera2\n"
            f"  USB Camera: {exc}"
        ) from exc


# ── GPIO helpers ──────────────────────────────────────────────────────────────

def _setup_button(pin: int) -> bool:
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        print(f"[GPIO] Button on GPIO {pin} (BCM, active-low).")
        return True
    except Exception as exc:
        print(f"[GPIO] GPIO unavailable ({exc}). Using keyboard Enter.")
        return False


def _button_pressed(pin: int) -> bool:
    try:
        import RPi.GPIO as GPIO
        return GPIO.input(pin) == GPIO.LOW
    except Exception:
        return False


def _cleanup_gpio():
    try:
        import RPi.GPIO as GPIO
        GPIO.cleanup()
    except Exception:
        pass


# ── Frame save helper ─────────────────────────────────────────────────────────

def _save_frame(frame, path: Path) -> None:
    import numpy as np
    from PIL import Image as PILImage
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(frame.astype(np.uint8)).convert("RGB").save(
        tmp := path.with_suffix(".tmp.jpg"), format="JPEG", quality=95
    )
    tmp.replace(path)
    print(f"[Save] {path}")


# ── Trigger / exit helpers ────────────────────────────────────────────────────

def _wait_for_trigger(gpio_ok: bool) -> None:
    if gpio_ok:
        while True:
            if _button_pressed(BUTTON_PIN):
                time.sleep(0.05)
                while _button_pressed(BUTTON_PIN):
                    time.sleep(0.02)
                return
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if ready:
                sys.stdin.readline()
                return
    else:
        try:
            input()
        except EOFError:
            pass


def _wait_for_exit():
    try:
        print("Press Enter to exit...")
        input()
    except (EOFError, KeyboardInterrupt):
        pass


# ── Result printer ────────────────────────────────────────────────────────────

def _print_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print("  MANGO GRADING RESULT")
    print("=" * 60)
    status = result.get("status", "UNKNOWN")
    grade  = result.get("predicted_grade")

    if status == "NON_MANGO" or not grade:
        # Validity model confirmed it's NOT a mango
        print("  Status  : INVALID – Not a mango")
        print("  Grade   : — (rejected by validity model)")
    else:
        conf   = result.get("grade_confidence", 0) * 100
        votes  = result.get("winning_votes", "?")
        vc     = result.get("vote_counts", {})
        avg    = result.get("average_model_scores", {})
        print(f"  Status     : {status}")
        print(f"  Grade      : {grade.title()}")
        print(f"  Confidence : {conf:.1f}%")
        print(f"  Votes      : {votes}/4 sides agreed")
        if status == "VALIDITY_UNAVAILABLE":
            print("  ⚠  Note    : Validity model not found — grade is based on")
            print("               the grade model only (install validity model")
            print("               for mango vs non-mango detection).")
        if vc:
            print("  Vote breakdown:")
            for g, v in vc.items():
                print(f"    {g.title():12s}: {v} vote(s)")
        if avg:
            print("  Avg scores (tied grades):")
            for g, s in avg.items():
                print(f"    {g.title():12s}: {s*100:.1f}%")

    sid = result.get("session_id", "")
    if sid:
        print(f"\n  Full report: {PROJECT_ROOT}/captures/{sid}/results.json")
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(camera_index: int = 0, use_display: bool = True, use_button: bool = True) -> None:
    print("\n" + "=" * 60)
    print("  MangoAI – Raspberry Pi Camera Grader")
    print("=" * 60)

    # Load models
    print("[Model] Loading grade model...")
    try:
        grade_model    = use_model.load_grade_model()
        validity_model = use_model.load_validity_model()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(2) from exc

    # OLED
    oled_device, oled_driver = None, None
    if use_display:
        oled_device, oled_driver = _load_oled()
        _oled_status(oled_device, oled_driver, "Ready", "Opening camera...")

    # Camera
    camera = _open_camera(camera_index)

    # GPIO button
    gpio_ok = _setup_button(BUTTON_PIN) if use_button else False

    # Session
    session_id  = uuid.uuid4().hex[:12]
    session_dir = PROJECT_ROOT / "captures" / session_id
    raw_dir     = session_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    captured_paths: dict[str, Path] = {}

    trig_msg = "GPIO button / Enter" if gpio_ok else "keyboard Enter"
    print(f"\n[Session] {session_id}")
    print(f"[Info]    Capture order: Right → Left → Front → Back")
    print(f"[Info]    Press {trig_msg} to capture each side.")
    print(f"[Info]    Ctrl-C to abort.\n")

    try:
        for side in CAMERA_SIDES:
            _oled_status(oled_device, oled_driver, f"Show {side.title()}", "Press button")
            print(f"[Capture] Show the {side.upper()} side, then press {trig_msg}... ",
                  end="", flush=True)
            _wait_for_trigger(gpio_ok)
            print("capturing...")
            time.sleep(CAPTURE_DELAY)

            frame      = camera.capture_frame()
            photo_path = raw_dir / f"{side}.jpg"
            _save_frame(frame, photo_path)
            captured_paths[side] = photo_path

            _oled_status(oled_device, oled_driver, f"{side.title()} saved", "")
            time.sleep(0.5)

        print("\n[Predict] Running 4-view model inference...")
        _oled_status(oled_device, oled_driver, "Predicting...", "Please wait")

        result = use_model.predict_captured_views(
            validity_model,
            grade_model,
            [captured_paths[s] for s in CAMERA_SIDES],
        )

        _print_result(result)
        _oled_result(oled_device, oled_driver, result)

        print("[Done] Result displayed on OLED. Press Enter or Ctrl-C to exit.")
        _wait_for_exit()

    except KeyboardInterrupt:
        print("\n[Aborted] Interrupted by user.")
    finally:
        _oled_clear(oled_device, oled_driver)
        camera.release()
        _cleanup_gpio()
        print("[Cleanup] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MangoAI – Raspberry Pi camera grader with OLED display.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--camera-index", type=int, default=0, metavar="N",
                        help="USB camera index (default 0). Ignored if Pi Camera is found.")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip OLED output (terminal only).")
    parser.add_argument("--no-button",  action="store_true",
                        help="Skip GPIO button; use keyboard Enter only.")
    args = parser.parse_args()
    run(camera_index=args.camera_index,
        use_display=not args.no_display,
        use_button=not args.no_button)
