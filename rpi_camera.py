"""
rpi_camera.py – Raspberry Pi mango grader with OLED display + voice.

Auto-detects camera:
  1. Raspberry Pi Camera Module via picamera2 (libcamera)
  2. USB webcam via OpenCV (cv2)

Captures 4 views (Right → Left → Front → Back) using a physical button
(GPIO 17, active-low) or keyboard Enter, then predicts the mango grade
using the trained model and shows the final result on a 1.3-inch
SSD1306 OLED display (128×64, I2C).

Non-mango fallback:
  When the validity model is not installed, objects whose grade-model
  confidence is below MIN_CONF_THRESHOLD are marked INVALID so that
  non-mangoes are not silently graded.

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
    python rpi_camera.py --no-voice
"""

from __future__ import annotations

import argparse
import queue
import select
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import use_model  # noqa: E402

CAMERA_SIDES = ("right", "left", "front", "back")
OLED_WIDTH   = 128
OLED_HEIGHT  = 64
BUTTON_PIN   = 17
CAPTURE_DELAY = 0.25

# ── If the validity model is absent, any side whose grade-model confidence
#    is below this threshold causes the whole result to be marked INVALID.
#    Tune between 0.30–0.50 depending on your test results.
MIN_CONF_THRESHOLD = 0.40


# ══════════════════════════════════════════════════════════════════════════════
# Voice helper (espeak-ng / espeak)
# ══════════════════════════════════════════════════════════════════════════════

class _Voice:
    """Non-blocking text-to-speech using espeak-ng (Linux / Raspberry Pi)."""

    def __init__(self) -> None:
        self.cmd = shutil.which("espeak-ng") or shutil.which("espeak")
        if self.cmd is None:
            print("[Voice] espeak-ng not found. Install: sudo apt install espeak-ng")
        else:
            print(f"[Voice] Using '{self.cmd}' for announcements.")

        self._q: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        if self.cmd:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def say(self, text: str) -> None:
        """Queue a message (latest replaces older unplayed ones)."""
        if self.cmd and not self._stop.is_set():
            # Drain old queued messages so only the latest plays
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._q.put(text)
            print(f"[Voice] {text}")

    def say_wait(self, text: str) -> None:
        """Speak and block until finished (used for important announcements)."""
        if self.cmd and not self._stop.is_set():
            print(f"[Voice] {text}")
            try:
                subprocess.run(
                    [self.cmd, text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                pass

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if text is None:
                return
            try:
                subprocess.run(
                    [self.cmd, text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                self.cmd = None
                return

    def stop(self) -> None:
        self._stop.set()
        self._q.put(None)


# ══════════════════════════════════════════════════════════════════════════════
# Non-mango fallback (confidence threshold)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_confidence_gate(result: dict) -> dict:
    """
    When the validity model is unavailable, use a confidence threshold to
    reject likely non-mango objects.

    If ANY per-side grade confidence is below MIN_CONF_THRESHOLD, override
    result status to NON_MANGO and clear the predicted grade.
    """
    if result.get("status") != "VALIDITY_UNAVAILABLE":
        return result   # validity model handled it already

    per_side = result.get("per_side", {})
    low_sides = []
    for side, side_res in per_side.items():
        conf = side_res.get("grade_confidence") or 0.0
        if conf < MIN_CONF_THRESHOLD:
            low_sides.append(f"{side}({conf*100:.0f}%)")

    if low_sides:
        print(f"[Gate] Low confidence on sides {low_sides} → marking INVALID")
        result = dict(result)
        result["status"]          = "NON_MANGO"
        result["predicted_grade"] = None
        result["rejection_reason"] = (
            f"confidence_below_threshold_{MIN_CONF_THRESHOLD}_on_{','.join(low_sides)}"
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# OLED helpers
# ══════════════════════════════════════════════════════════════════════════════

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
    """Render lines on OLED. First line uses large bold font, rest use small."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    image = PILImage.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)
    draw  = ImageDraw.Draw(image)

    # Try bold large font for headline
    large_font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            large_font = ImageFont.truetype(font_path, 20)
            break
        except Exception:
            continue
    if large_font is None:
        large_font = ImageFont.load_default()

    small_font = ImageFont.load_default()

    y = 2
    for i, line in enumerate(lines):
        font = large_font if i == 0 else small_font
        lh   = 22 if i == 0 else 13
        draw.text((2, y), line, fill=255, font=font)
        y += lh

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
    """Show a 2-line status on OLED (both small font)."""
    if device is None:
        return
    _oled_render(device, driver, [l1, l2])


def _oled_result(device, driver: str, result: dict) -> None:
    """Show clean result: Grade + Confidence, or INVALID."""
    if device is None:
        return
    status = result.get("status", "")
    grade  = result.get("predicted_grade")

    if status == "NON_MANGO" or not grade:
        # Large INVALID, small explanation
        _oled_render(device, driver, ["INVALID", "Not a mango"])
    else:
        # Get confidence from grade_confidence or average_model_scores
        conf = result.get("grade_confidence")
        if conf is None:
            avg_scores = result.get("average_model_scores", {})
            conf = avg_scores.get(grade, 0)
        conf_pct = conf * 100
        _oled_render(device, driver, [
            grade.title(),           # e.g. "Grade 1"   ← large
            f"Conf: {conf_pct:.1f}%",  # e.g. "Conf: 87.3%" ← small
        ])


# ══════════════════════════════════════════════════════════════════════════════
# Camera helpers
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# GPIO helpers
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Frame save helper
# ══════════════════════════════════════════════════════════════════════════════

def _save_frame(frame, path: Path) -> None:
    import numpy as np
    from PIL import Image as PILImage
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(frame.astype(np.uint8)).convert("RGB").save(
        tmp := path.with_suffix(".tmp.jpg"), format="JPEG", quality=95
    )
    tmp.replace(path)
    print(f"[Save] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Trigger / exit helpers
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Result printer
# ══════════════════════════════════════════════════════════════════════════════

def _print_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print("  MANGO GRADING RESULT")
    print("=" * 60)
    status = result.get("status", "UNKNOWN")
    grade  = result.get("predicted_grade")

    if status == "NON_MANGO" or not grade:
        print("  Status  : INVALID – Not a mango")
        reason = result.get("rejection_reason", "")
        if "confidence_below_threshold" in reason:
            print(f"  Reason  : Low confidence (threshold={MIN_CONF_THRESHOLD*100:.0f}%)")
            print("            Object is likely not a mango.")
        else:
            print("  Reason  : Rejected by validity model.")
    else:
        # Get confidence
        conf = result.get("grade_confidence")
        if conf is None:
            avg = result.get("average_model_scores", {})
            conf = avg.get(grade, 0)
        conf_pct = conf * 100
        votes = result.get("winning_votes", "?")
        vc    = result.get("vote_counts", {})

        print(f"  Status     : {status}")
        print(f"  Grade      : {grade.title()}")
        print(f"  Confidence : {conf_pct:.1f}%")
        print(f"  Votes      : {votes}/4 sides agreed")
        if status == "VALIDITY_UNAVAILABLE":
            print(f"  ⚠  Note  : Validity model not found. Confidence gate")
            print(f"             ({MIN_CONF_THRESHOLD*100:.0f}%) was used as fallback.")
        if vc:
            print("  Vote breakdown:")
            for g, v in vc.items():
                print(f"    {g.title():12s}: {v} vote(s)")

    sid = result.get("session_id", "")
    if sid:
        print(f"\n  Full report: {PROJECT_ROOT}/captures/{sid}/results.json")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run(
    camera_index: int = 0,
    use_display: bool = True,
    use_button: bool = True,
    use_voice: bool = True,
) -> None:
    print("\n" + "=" * 60)
    print("  MangoAI – Raspberry Pi Camera Grader")
    print("=" * 60)

    # ── Voice ──────────────────────────────────────────────────────────────
    voice = _Voice() if use_voice else None

    # ── Load models ────────────────────────────────────────────────────────
    print("[Model] Loading grade model...")
    if voice:
        voice.say("Loading model, please wait.")
    try:
        grade_model    = use_model.load_grade_model()
        validity_model = use_model.load_validity_model()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(2) from exc

    if validity_model is None:
        print(f"[Gate] Validity model not found. "
              f"Using confidence threshold ({MIN_CONF_THRESHOLD*100:.0f}%) "
              f"to reject non-mangoes.")

    # ── OLED ───────────────────────────────────────────────────────────────
    oled_device, oled_driver = None, None
    if use_display:
        oled_device, oled_driver = _load_oled()
        _oled_status(oled_device, oled_driver, "MangoAI Ready", "Opening camera...")

    # ── Camera ─────────────────────────────────────────────────────────────
    camera = _open_camera(camera_index)

    # ── GPIO button ────────────────────────────────────────────────────────
    gpio_ok = _setup_button(BUTTON_PIN) if use_button else False

    # ── Session ────────────────────────────────────────────────────────────
    session_id  = uuid.uuid4().hex[:12]
    session_dir = PROJECT_ROOT / "captures" / session_id
    raw_dir     = session_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    captured_paths: dict[str, Path] = {}

    trig_msg = "GPIO button or Enter" if gpio_ok else "keyboard Enter"
    print(f"\n[Session] {session_id}")
    print(f"[Info]    Capture order: Right → Left → Front → Back")
    print(f"[Info]    Press {trig_msg} to capture each side.")
    print(f"[Info]    Ctrl-C to abort.\n")

    if voice:
        voice.say_wait("Mango grader ready. Place the mango and press the button.")

    try:
        # ── 4-side capture loop ────────────────────────────────────────────
        for i, side in enumerate(CAMERA_SIDES):
            # OLED: prompt
            _oled_status(oled_device, oled_driver,
                         f"Show {side.title()} side", "Press button")

            # Voice: prompt user for this side
            if voice:
                voice.say_wait(f"Show the {side} side and press the button.")

            print(f"[Capture] Show the {side.upper()} side, "
                  f"then press {trig_msg}... ", end="", flush=True)

            _wait_for_trigger(gpio_ok)
            print("capturing...")

            # Voice: acknowledge click immediately
            if voice:
                voice.say(f"{side} side, capturing.")

            time.sleep(CAPTURE_DELAY)

            frame      = camera.capture_frame()
            photo_path = raw_dir / f"{side}.jpg"
            _save_frame(frame, photo_path)
            captured_paths[side] = photo_path

            # OLED + voice: confirm saved
            _oled_status(oled_device, oled_driver,
                         f"{side.title()} saved", f"{i+1}/4 done")
            if voice:
                voice.say(f"{side} side saved. {4 - i - 1} remaining.")
            time.sleep(0.4)

        # ── Predict ────────────────────────────────────────────────────────
        print("\n[Predict] Running 4-view model inference...")
        _oled_status(oled_device, oled_driver, "Predicting...", "Please wait")
        if voice:
            voice.say_wait("All four sides captured. Predicting now, please wait.")

        result = use_model.predict_captured_views(
            validity_model,
            grade_model,
            [captured_paths[s] for s in CAMERA_SIDES],
        )

        # Apply confidence gate if validity model was not available
        result = _apply_confidence_gate(result)

        # ── Show result ────────────────────────────────────────────────────
        _print_result(result)
        _oled_result(oled_device, oled_driver, result)

        # Voice announcement of final result
        if voice:
            grade = result.get("predicted_grade")
            status = result.get("status", "")
            if status == "NON_MANGO" or not grade:
                voice.say_wait("Result: Invalid. This is not a mango.")
            else:
                conf = result.get("grade_confidence")
                if conf is None:
                    avg = result.get("average_model_scores", {})
                    conf = avg.get(grade, 0)
                conf_pct = int(conf * 100)
                voice.say_wait(
                    f"Result: {grade}. Confidence {conf_pct} percent."
                )

        print("[Done] Result on OLED. Press Enter or Ctrl-C to exit.")
        _wait_for_exit()

    except KeyboardInterrupt:
        print("\n[Aborted] Interrupted by user.")
        if voice:
            voice.say("Aborted.")
    finally:
        _oled_clear(oled_device, oled_driver)
        camera.release()
        _cleanup_gpio()
        if voice:
            voice.stop()
        print("[Cleanup] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MangoAI – Raspberry Pi camera grader with OLED + voice.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--camera-index", type=int, default=0, metavar="N",
                        help="USB camera index (default 0). Ignored if Pi Camera found.")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip OLED output (terminal only).")
    parser.add_argument("--no-button", action="store_true",
                        help="Skip GPIO button; use keyboard Enter only.")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable voice announcements.")
    args = parser.parse_args()
    run(
        camera_index=args.camera_index,
        use_display=not args.no_display,
        use_button=not args.no_button,
        use_voice=not args.no_voice,
    )
