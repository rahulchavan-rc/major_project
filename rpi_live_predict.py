"""
rpi_live_predict.py – MangoAI LIVE predictor for Raspberry Pi.

Combines the real-time KNN prediction from live_predict.py with the
Raspberry Pi hardware support from rpi_camera.py:
  • Camera  : picamera2 (Pi Camera Module) or USB webcam (OpenCV fallback)
  • Display : SSD1306 OLED (128×64, I2C, luma.oled or Adafruit) — optional
  • Button  : GPIO 17 (active-low) — optional
  • Voice   : espeak-ng / espeak (Linux / Raspberry Pi)

LIVE MODE
──────────
Continuously grabs frames from the Pi camera and runs KNN cosine-similarity
prediction every N frames. The result is shown on the OLED and (optionally)
the OpenCV preview window.

VOICE COMMANDS (GPIO button, GPIO 17)
──────────────────────────────────────
    Short press  (<1 s)          : Save current frame and announce its prediction
  Long  press  (≥1 s)          : Freeze / unfreeze prediction updates
    Double press (≤0.6 s gap)    : Save current frame to disk

KEYBOARD CONTROLS (OpenCV window)
───────────────────────────────────
  Q / ESC  – Quit
  SPACE    – Freeze / unfreeze
  S        – Save current frame
  +  / =   – Speed up  predictions (fewer frames to skip)
  -        – Slow  down predictions
  F        – Toggle verbose console output
  V        – Toggle voice on/off
  A        – Announce current grade (voice)

OUTPUT FORMAT  (identical to live_predict.py)
──────────────────────────────────────────────
  • Top-bar  : grade label + confidence %
  • Side-bar : KNN vote bars for each grade class
  • OLED     : grade label + confidence on 2-line display
  • Console  : [Predict] Grade 2  (80%)  sim=0.923 …

Usage:
    python rpi_live_predict.py
    python rpi_live_predict.py --every 10 --k 7
    python rpi_live_predict.py --no-display   # headless / no OpenCV window
    python rpi_live_predict.py --no-oled      # skip OLED
    python rpi_live_predict.py --no-button    # keyboard only
    python rpi_live_predict.py --no-voice     # silent mode
    python rpi_live_predict.py --camera-index 1
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PATH = PROJECT_ROOT / "extracted_features.json"
CAPTURES_DIR  = PROJECT_ROOT / "captures" / "rpi_live"

# ── Grade label mapping ───────────────────────────────────────────────────────
GRADE_KEYWORDS = {
    "grade 1": "Grade 1",
    "grade 2": "Grade 2",
    "grade 3": "Grade 3",
    "grade 4": "Grade 4",
}

# ── Confidence tiers ──────────────────────────────────────────────────────────
CONF_TIERS = [
    (0.80, "very high"),
    (0.60, "high"),
    (0.40, "medium"),
    (0.00, "low"),
]

# ── Hardware constants ────────────────────────────────────────────────────────
OLED_WIDTH           = 128
OLED_HEIGHT          = 64
BUTTON_PIN           = 17
DOUBLE_PRESS_WINDOW  = 0.6    # seconds — max gap for a double-press
LONG_PRESS_THRESHOLD = 1.0    # seconds — hold time for long-press
RESULT_DISPLAY_SECONDS = 3.0

# ── OpenCV colours (BGR) ──────────────────────────────────────────────────────
CLR_RED    = (60,  60,  240)
CLR_YELLOW = (0,   200, 255)
CLR_WHITE  = (255, 255, 255)
CLR_DARK   = (20,  20,  20)
CLR_NEON   = (100, 255, 100)
CLR_ORANGE = (0,   165, 255)
CLR_CYAN   = (255, 220, 0)


# ══════════════════════════════════════════════════════════════════════════════
# Voice helper  (espeak-ng / espeak — Linux / Raspberry Pi)
# ══════════════════════════════════════════════════════════════════════════════

class _Voice:
    """Non-blocking TTS using espeak-ng or espeak."""

    def __init__(self) -> None:
        self.cmd = shutil.which("espeak-ng") or shutil.which("espeak")
        if self.cmd is None:
            print("[Voice] espeak-ng not found. Install: sudo apt install espeak-ng")
        else:
            print(f"[Voice] Using '{self.cmd}' for announcements.")

        self._q: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()

        if self.cmd:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    # ── public ----------------------------------------------------------------
    def say(self, text: str) -> None:
        """Queue a message — latest replaces older unplayed ones."""
        if self.cmd and not self._stop.is_set():
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._q.put(text)
            print(f"[Voice] {text}")

    def say_wait(self, text: str) -> None:
        """Speak and BLOCK until finished (for important announcements)."""
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

    def stop(self) -> None:
        self._stop.set()
        self._q.put(None)
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()

    # ── private ---------------------------------------------------------------
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if text is None:
                return
            try:
                with self._lock:
                    if self._stop.is_set():
                        return
                    self._proc = subprocess.Popen(
                        [self.cmd, text],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                self._proc.wait()
            except OSError:
                self.cmd = None
                return
            finally:
                with self._lock:
                    self._proc = None


# ══════════════════════════════════════════════════════════════════════════════
# OLED helpers  (luma.oled  or  Adafruit_SSD1306)
# ══════════════════════════════════════════════════════════════════════════════

def _load_oled():
    """Try luma.oled first, then Adafruit_SSD1306."""
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

    print("[OLED] No OLED display found (luma.oled / Adafruit_SSD1306 not installed).")
    return None, None


# ── cached fonts (loaded once) ────────────────────────────────────────────────
_OLED_FONTS: dict = {}


def _get_oled_fonts() -> tuple:
    """Return (large_font, small_font) — loaded once and cached."""
    if _OLED_FONTS:
        return _OLED_FONTS["large"], _OLED_FONTS["small"]

    from PIL import ImageFont
    large = None
    for size, paths in [
        (16, [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]),
    ]:
        for fp in paths:
            try:
                large = ImageFont.truetype(fp, size)
                break
            except Exception:
                continue
        if large:
            break
    if large is None:
        large = ImageFont.load_default()

    small = ImageFont.load_default()
    _OLED_FONTS["large"] = large
    _OLED_FONTS["small"] = small
    return large, small


def _oled_push(device, driver: str, image) -> None:
    """Push a PIL image to the OLED device."""
    if driver == "luma":
        device.display(image)
    elif driver == "adafruit":
        device.image(image)
        device.display()


def _oled_render(device, driver: str, lines: list[str]) -> None:
    """Render plain text lines on the OLED (line 0 = large bold, rest = small)."""
    from PIL import Image as PILImage, ImageDraw
    large_font, small_font = _get_oled_fonts()
    image = PILImage.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)
    draw  = ImageDraw.Draw(image)
    y = 2
    for i, line in enumerate(lines):
        font = large_font if i == 0 else small_font
        lh   = 19 if i == 0 else 11
        draw.text((2, y), line, fill=255, font=font)
        y += lh
    _oled_push(device, driver, image)


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
    _oled_render(device, driver, [l1, l2])


def _oled_live(
    device,
    driver: str,
    label: str,
    confidence: float,
    top_k: list[tuple[str, float]],
    frozen: bool,
) -> None:
    """
    Full-screen OLED render — mirrors the live_predict.py OpenCV overlay:

    ┌────────────────────────────┐
    │ GRADE 2          [FROZEN]  │  ← large bold (row 0, y=0..18)
    │ Conf: 80% ████████░░░░░░  │  ← confidence bar (y=20)
    │ ─────────────────────────  │  ← divider
    │ G1 ██░░  0   G3 ░░░░  0  │  ← vote bars row 1 (y=33)
    │ G2 ████  3   G4 ░░░░  0  │  ← vote bars row 2 (y=44)
    │ INV░░░░  0               │  ← vote bars row 3 (y=55)
    └────────────────────────────┘
    """
    if device is None:
        return

    from PIL import Image as PILImage, ImageDraw
    from collections import Counter as _Counter

    large_font, small_font = _get_oled_fonts()
    img  = PILImage.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)
    draw = ImageDraw.Draw(img)

    W, H = OLED_WIDTH, OLED_HEIGHT  # 128, 64

    # ── Row 0: Grade label (large) ─────────────────────────────────────────
    is_invalid = label == "invalid"
    initialising = label in ("Initialising...", "")

    if initialising:
        draw.text((2, 0), "Initialising...", fill=255, font=small_font)
        _oled_push(device, driver, img)
        return

    if is_invalid:
        grade_str = "INVALID"
    else:
        grade_str = label.upper()          # e.g. "GRADE 2"

    draw.text((2, 1), grade_str, fill=255, font=large_font)

    if frozen:
        draw.text((W - 28, 1), "[FRZ]", fill=255, font=small_font)

    # ── Row 1: Confidence % + filled bar ──────────────────────────────────
    conf_y  = 20
    conf_pct = int(round(confidence * 100))
    conf_str = f"Conf:{conf_pct}%"
    draw.text((2, conf_y), conf_str, fill=255, font=small_font)

    # Bar: from x=46 to x=126 (80 px wide)
    bar_x0, bar_x1, bar_y0, bar_y1 = 46, 126, conf_y + 1, conf_y + 8
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], outline=1, fill=0)
    fill_w = int((bar_x1 - bar_x0) * confidence)
    if fill_w > 0:
        draw.rectangle([bar_x0, bar_y0, bar_x0 + fill_w, bar_y1], fill=1)

    # ── Divider ────────────────────────────────────────────────────────────
    div_y = conf_y + 11
    draw.line([(0, div_y), (W - 1, div_y)], fill=1)

    # ── KNN vote bars — 2 columns, 3 rows ──────────────────────────────────
    # Layout (each cell = 64 px wide):
    #   col 0 (x=0):  G1 / G2 / INV
    #   col 1 (x=64): G3 / G4
    vote_counts = _Counter(lbl for lbl, _ in top_k)
    k_total     = max(sum(vote_counts.values()), 1)

    all_grades  = [("G1", "Grade 1"), ("G2", "Grade 2"),
                   ("G3", "Grade 3"), ("G4", "Grade 4"), ("INV", "invalid")]

    # Two-column grid:
    #   left col  (x offset 0)  → G1, G2, INV
    #   right col (x offset 64) → G3, G4
    layout = [
        (0,  div_y + 3,  "G1",  "Grade 1"),
        (0,  div_y + 13, "G2",  "Grade 2"),
        (64, div_y + 3,  "G3",  "Grade 3"),
        (64, div_y + 13, "G4",  "Grade 4"),
        (0,  div_y + 23, "INV", "invalid"),
    ]

    bar_max = 38   # max bar width in pixels per column
    for col_x, row_y, short, grade_key in layout:
        cnt   = vote_counts.get(grade_key, 0)
        frac  = cnt / k_total
        is_winner = (grade_key == label)

        # Label (e.g. "G2")
        draw.text((col_x, row_y), short, fill=255, font=small_font)

        # Mini bar track (18..56 relative to col_x)
        bx0 = col_x + 18
        bx1 = col_x + 18 + bar_max
        by0 = row_y + 1
        by1 = row_y + 7
        draw.rectangle([bx0, by0, bx1, by1], outline=1, fill=0)
        if frac > 0:
            fw = max(1, int(bar_max * frac))
            draw.rectangle([bx0, by0, bx0 + fw, by1], fill=1)

        # Vote count
        draw.text((bx1 + 2, row_y), str(cnt), fill=255, font=small_font)

    _oled_push(device, driver, img)


# ══════════════════════════════════════════════════════════════════════════════
# GPIO helpers
# ══════════════════════════════════════════════════════════════════════════════

def _setup_button(pin: int) -> bool:
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        print(f"[GPIO] Button on GPIO {pin} (BCM, active-low, pull-up).")
        return True
    except Exception as exc:
        print(f"[GPIO] GPIO unavailable ({exc}). Button commands disabled.")
        return False


def _button_pressed(pin: int) -> bool:
    try:
        import RPi.GPIO as GPIO
        return GPIO.input(pin) == GPIO.LOW
    except Exception:
        return False


def _cleanup_gpio() -> None:
    try:
        import RPi.GPIO as GPIO
        GPIO.cleanup()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Camera helpers  (picamera2  →  USB webcam fallback)
# ══════════════════════════════════════════════════════════════════════════════

class _PiCamera:
    """Raspberry Pi Camera Module via picamera2 (libcamera)."""

    def __init__(self) -> None:
        from picamera2 import Picamera2
        self._cam = Picamera2()
        cfg = self._cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
        )
        self._cam.configure(cfg)
        self._cam.start()
        time.sleep(2.0)
        print("[Camera] Raspberry Pi Camera Module (picamera2) ready.")

    def read(self):
        """Return (True, RGB frame) — mirrors cv2.VideoCapture.read()."""
        return True, self._cam.capture_array("main")

    def release(self) -> None:
        self._cam.stop()
        self._cam.close()


class _UsbCamera:
    """USB webcam via OpenCV — fallback when picamera2 is unavailable."""

    def __init__(self, index: int) -> None:
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"Cannot open USB camera index {index}.")
        for _ in range(5):
            self._cap.read()
        print(f"[Camera] USB webcam at index {index} (OpenCV) ready.")

    def read(self):
        """Return (ok, RGB frame)."""
        ok, frame = self._cap.read()
        if not ok:
            return False, None
        return True, self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        self._cap.release()


def _open_camera(camera_index: int):
    try:
        return _PiCamera()
    except Exception as exc:
        print(f"[Camera] picamera2 unavailable ({exc}), trying USB webcam…")
    try:
        return _UsbCamera(camera_index)
    except Exception as exc:
        raise RuntimeError(
            "Could not open any camera.\n"
            "  Pi Camera : sudo apt install python3-picamera2\n"
            f"  USB Camera: {exc}"
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# Feature database
# ══════════════════════════════════════════════════════════════════════════════

def _parse_grade_from_path(image_path: str) -> str:
    lower = image_path.lower()
    for kw, label in GRADE_KEYWORDS.items():
        if kw in lower:
            return label
    return "Unknown"


def load_feature_db(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load extracted_features.json.
    Returns:
        features   – (N, D) float32 L2-normalised array
        labels     – list of 'Grade 1'…'Grade 4' or 'invalid'
        img_paths  – list of original image paths
    """
    print(f"[DB] Loading feature database from {path.name}…")
    with path.open("r", encoding="utf-8") as f:
        db = json.load(f)

    samples   = db["samples"]
    raw_feats = np.array([s["features"] for s in samples], dtype=np.float32)
    norms     = np.linalg.norm(raw_feats, axis=1, keepdims=True)
    norms     = np.where(norms == 0, 1.0, norms)
    features  = raw_feats / norms

    labels = []
    for s in samples:
        if s["label"] == "invalid":
            labels.append("invalid")
        else:
            labels.append(_parse_grade_from_path(s["image_path"]))

    img_paths    = [s["image_path"] for s in samples]
    grade_counts = Counter(l for l in labels if l != "invalid")
    print(
        f"[DB] {len(samples)} samples | "
        f"valid: {sum(1 for l in labels if l != 'invalid')} | "
        f"invalid: {sum(1 for l in labels if l == 'invalid')}"
    )
    print(f"[DB] Grade distribution: {dict(grade_counts)}")
    return features, labels, img_paths


# ══════════════════════════════════════════════════════════════════════════════
# CNN feature extractor  (same backbone as training / live_predict.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_extractor():
    """
    Priority:
      1. Tap 'cnn_features' layer of the trained project model.
      2. Fresh MobileNetV3Small with ImageNet weights as fallback.
    """
    import tensorflow as tf
    import use_model

    GRADE_CANDIDATES = [
        PROJECT_ROOT / "mobilenetv3_runs" / "best_model.keras",
        PROJECT_ROOT / "mobilenetv3_runs" / "final_model.keras",
        PROJECT_ROOT.parent / "pro copy 2" / "mobilenetv3_runs" / "best_model.keras",
        PROJECT_ROOT.parent / "pro copy 2" / "mobilenetv3_runs" / "final_model.keras",
    ]
    grade_path = next((p for p in GRADE_CANDIDATES if p.exists()), None)

    if grade_path:
        print(f"[Model] Loading trained model: {grade_path.name}")
        try:
            full_model = use_model.load_grade_model()
            feat_layer = full_model.get_layer("cnn_features")
            image_input = next(
                (
                    model_input
                    for model_input in full_model.inputs
                    if model_input.name.split(":")[0].split("/")[0] == "image"
                ),
                None,
            )
            if image_input is None:
                if len(full_model.inputs) != 1:
                    raise ValueError("Could not find the model's named 'image' input.")
                image_input = full_model.inputs[0]

            feature_dimension = int(feat_layer.output.shape[-1])
            if feature_dimension != 576:
                raise ValueError(
                    f"Expected 576 cnn_features, but the model provides {feature_dimension}."
                )
            extractor = tf.keras.Model(inputs=image_input, outputs=feat_layer.output)
            print(f"[Model] Extractor → layer '{feat_layer.name}' "
                  f"shape {feat_layer.output.shape}")
            return extractor
        except Exception as exc:
            raise RuntimeError(
                f"Could not build the 576-dimension cnn_features extractor from "
                f"{grade_path.name}: {exc}"
            ) from exc

    print("[Model] Building fresh MobileNetV3-Small feature extractor…")
    from tensorflow.keras.applications import MobileNetV3Small
    base = MobileNetV3Small(input_shape=(224, 224, 3), include_top=False,
                            weights="imagenet", pooling="avg")
    base.trainable = False
    extractor = tf.keras.Model(inputs=base.input, outputs=base.output)
    print(f"[Model] Extractor ready — output shape: {extractor.output_shape}")
    return extractor


def extract_features(model, frame_rgb: np.ndarray) -> np.ndarray:
    """Resize to 224×224, supply all model inputs, and L2-normalise features."""
    import tensorflow as tf
    import use_model

    img   = tf.image.resize(tf.constant(frame_rgb, dtype=tf.float32),
                            [224, 224], method="bilinear")
    img   = tf.expand_dims(img, 0)
    model_inputs = use_model.grade_model_inputs(model, img)
    feats = model(model_inputs, training=False).numpy()[0]
    norm  = np.linalg.norm(feats)
    return feats / norm if norm > 0 else feats


# ══════════════════════════════════════════════════════════════════════════════
# KNN prediction
# ══════════════════════════════════════════════════════════════════════════════

def knn_predict(
    query: np.ndarray,
    db_features: np.ndarray,
    db_labels: list[str],
    k: int = 5,
) -> tuple[str, float, list[tuple[str, float]]]:
    """
    Cosine-similarity KNN.
    Returns:
        predicted_label, confidence (0-1), top_k list of (label, similarity)
    """
    if db_features.ndim != 2 or query.ndim != 1 or db_features.shape[1] != query.shape[0]:
        database_dimension = db_features.shape[1] if db_features.ndim == 2 else "invalid"
        raise ValueError(
            f"Feature dimension mismatch: database has {database_dimension}, "
            f"live extractor produced {query.shape[0]}-D features."
        )
    sims      = db_features @ query
    top_k_idx = np.argpartition(sims, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]
    top_k     = [(db_labels[i], float(sims[i])) for i in top_k_idx]
    votes     = Counter(label for label, _ in top_k)
    winner    = votes.most_common(1)[0][0]
    confidence = votes[winner] / k
    return winner, confidence, top_k


# ══════════════════════════════════════════════════════════════════════════════
# Speech helpers
# ══════════════════════════════════════════════════════════════════════════════

def _confidence_tier(confidence: float) -> str:
    for threshold, label in CONF_TIERS:
        if confidence >= threshold:
            return label
    return "low"


def _build_speech_message(label: str, confidence: float) -> str:
    """Build a voice announcement identical to live_predict.py."""
    pct  = int(round(confidence * 100))
    tier = _confidence_tier(confidence)
    if label == "invalid":
        return f"Invalid. Not a mango. Confidence {pct} percent."
    return f"{label}. {tier} confidence. {pct} percent."


# ══════════════════════════════════════════════════════════════════════════════
# OpenCV overlay  (identical style to live_predict.py)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_overlay(
    frame,
    cv2,
    label: str,
    confidence: float,
    top_k: list[tuple[str, float]],
    fps: float,
    frozen: bool,
    every: int,
    gpio_ok: bool,
    voice_on: bool,
) -> None:
    h, w = frame.shape[:2]

    # ── Top dark bar ───────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), CLR_DARK, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    is_invalid   = label == "invalid"
    grade_colour = CLR_RED if is_invalid else CLR_NEON
    grade_text   = "INVALID - Not a mango" if is_invalid else label.upper()

    cv2.putText(frame, grade_text,
                (14, 38), cv2.FONT_HERSHEY_DUPLEX, 1.1,
                grade_colour, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{confidence*100:.0f}% confidence",
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                CLR_WHITE, 1, cv2.LINE_AA)

    # ── RPi / Voice badge (top-right) ──────────────────────────────────────
    badge = f"RPi {'BTN ' if gpio_ok else ''}| Voice:{'ON' if voice_on else 'OFF'}"
    cv2.putText(frame, badge,
                (w - len(badge) * 7 - 8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, CLR_CYAN, 1, cv2.LINE_AA)

    # ── KNN vote bars (right side) — identical to live_predict.py ─────────
    vote_counts = Counter(lbl for lbl, _ in top_k)
    all_grades  = ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "invalid"]
    bar_x, bar_y0, bw, bh = w - 160, 85, 130, 14

    cv2.rectangle(frame,
                  (bar_x - 8, bar_y0 - 20),
                  (w - 4,     bar_y0 + len(all_grades) * 24 + 4),
                  CLR_DARK, -1)

    for i, g in enumerate(all_grades):
        cnt  = vote_counts.get(g, 0)
        frac = cnt / max(sum(vote_counts.values()), 1)
        y    = bar_y0 + i * 24
        col  = CLR_NEON if g == label else (100, 100, 100)
        short = g.replace("Grade ", "G") if "Grade" in g else "INV"
        cv2.putText(frame, short, (bar_x - 8, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x + 22, y), (bar_x + 22 + bw, y + bh),
                      (50, 50, 50), -1)
        if frac > 0:
            cv2.rectangle(frame, (bar_x + 22, y),
                          (bar_x + 22 + int(bw * frac), y + bh), col, -1)
        cv2.putText(frame, f"{cnt}", (bar_x + 22 + bw + 4, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    # ── Frozen banner ──────────────────────────────────────────────────────
    if frozen:
        cv2.putText(frame, "[ FROZEN ]",
                    (w // 2 - 55, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, CLR_ORANGE, 2, cv2.LINE_AA)

    # ── Bottom status ──────────────────────────────────────────────────────
    status = []
    if frozen:
        status.append("FROZEN")
    status.append(f"FPS:{fps:.1f}")
    status.append(f"every:{every}fr")
    cv2.putText(frame, "  ".join(status),
                (8, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_YELLOW, 1, cv2.LINE_AA)

    kb_line = "Q=quit  S=save  SPACE=freeze  +/-=speed  V=voice  A=announce  F=info"
    if gpio_ok:
        kb_line = "BTN:short=save+announce long=freeze dbl=save  |  " + kb_line
    cv2.putText(frame, kb_line,
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.28, CLR_YELLOW, 1, cv2.LINE_AA)

    # ── Watermark ──────────────────────────────────────────────────────────
    cv2.putText(frame, "MangoAI-RPi",
                (w - 90, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (60, 60, 60), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# GPIO button event classifier  (short / long / double press)
# ══════════════════════════════════════════════════════════════════════════════

class _ButtonMonitor:
    """
    Background thread that monitors GPIO pin and classifies button presses:
      short  – single press  < LONG_PRESS_THRESHOLD s
      long   – hold          >= LONG_PRESS_THRESHOLD s
      double – two short presses within DOUBLE_PRESS_WINDOW s
    """

    def __init__(self, pin: int) -> None:
        self._pin   = pin
        self._event: str | None = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def get_event(self) -> str | None:
        with self._lock:
            ev, self._event = self._event, None
        return ev

    def _set(self, ev: str) -> None:
        with self._lock:
            self._event = ev
        print(f"[BTN] {ev.upper()} press detected.")

    def _run(self) -> None:
        last_release: float | None = None
        while not self._stop.is_set():
            if not _button_pressed(self._pin):
                time.sleep(0.02)
                continue

            press_start = time.time()
            while _button_pressed(self._pin) and not self._stop.is_set():
                time.sleep(0.02)
            hold = time.time() - press_start

            if hold >= LONG_PRESS_THRESHOLD:
                self._set("long")
                last_release = None
                continue

            now = time.time()
            if last_release is not None and (now - last_release) <= DOUBLE_PRESS_WINDOW:
                self._set("double")
                last_release = None
            else:
                last_release = now
                deadline = now + DOUBLE_PRESS_WINDOW
                pressed_again = False
                while time.time() < deadline and not self._stop.is_set():
                    time.sleep(0.05)
                    if _button_pressed(self._pin):
                        pressed_again = True
                        break
                if not pressed_again:
                    self._set("short")
                    last_release = None

            time.sleep(0.05)

    def stop(self) -> None:
        self._stop.set()


# ══════════════════════════════════════════════════════════════════════════════
# Frame save helper
# ══════════════════════════════════════════════════════════════════════════════

def _save_frame_rgb(frame_rgb: np.ndarray, path: Path, cv2=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image as PILImage
        PILImage.fromarray(frame_rgb.astype(np.uint8)).convert("RGB").save(
            str(path), format="JPEG", quality=95
        )
    except Exception:
        if cv2 is not None:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), frame_bgr)
    print(f"[Save] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main live-predict loop
# ══════════════════════════════════════════════════════════════════════════════

def run(
    camera_index: int  = 0,
    k: int             = 5,
    every: int         = 15,
    use_display: bool  = True,   # OpenCV preview window
    use_oled: bool     = True,   # SSD1306 OLED
    use_button: bool   = True,   # GPIO button
    use_voice: bool    = True,   # espeak TTS
) -> None:

    # ── Load feature database ──────────────────────────────────────────────
    if not FEATURES_PATH.exists():
        raise SystemExit(f"[ERROR] Feature database not found: {FEATURES_PATH}")
    db_features, db_labels, _ = load_feature_db(FEATURES_PATH)

    # ── Build CNN feature extractor ────────────────────────────────────────
    extractor = build_feature_extractor()

    # ── Voice ──────────────────────────────────────────────────────────────
    voice = _Voice() if use_voice else None

    # ── OLED ───────────────────────────────────────────────────────────────
    oled_device, oled_driver = (None, None)
    if use_oled:
        oled_device, oled_driver = _load_oled()

    _oled_status(oled_device, oled_driver, "MangoAI Live", "Opening camera...")

    # ── Camera ─────────────────────────────────────────────────────────────
    camera = _open_camera(camera_index)

    # ── GPIO button ────────────────────────────────────────────────────────
    gpio_ok  = _setup_button(BUTTON_PIN) if use_button else False
    btn_mon: _ButtonMonitor | None = _ButtonMonitor(BUTTON_PIN) if gpio_ok else None

    # ── OpenCV window ──────────────────────────────────────────────────────
    cv2 = None
    window_name = "MangoAI - RPi Live Prediction  (Q to quit)"
    if use_display:
        try:
            import cv2 as _cv2
            cv2 = _cv2
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 520)
        except ImportError:
            print("[WARN] opencv-python not installed — running headless.")
            cv2 = None

    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Banner ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  MangoAI – RPi Live Camera Predictor")
    print("=" * 62)
    print(f"  K-nearest   : {k}")
    print(f"  Predict every {every} frames")
    print(f"  Result hold : {RESULT_DISPLAY_SECONDS:g} seconds")
    print(f"  Voice       : {'ON' if use_voice else 'OFF'}")
    print(f"  OLED        : {'ON' if oled_device else 'OFF / not found'}")
    print(f"  GPIO button : {'ON  (GPIO ' + str(BUTTON_PIN) + ')' if gpio_ok else 'OFF'}")
    if gpio_ok:
        print(f"  BTN cmds    : short=save+announce  long=freeze  double=save")
    print(f"  Keyboard    : Q=quit  SPACE=freeze  S=save  +/-=speed  V=voice  A=announce")
    print("=" * 62 + "\n")

    if voice:
        voice.say("MangoAI live predictor ready.")

    _oled_status(oled_device, oled_driver, "MangoAI Ready", "Predicting live...")

    # ── State variables ────────────────────────────────────────────────────
    label       = "Initialising..."
    confidence  = 0.0
    top_k: list[tuple[str, float]] = []
    frozen      = False
    show_info   = False
    voice_on    = use_voice
    frame_num   = 0
    fps         = 0.0
    t_prev      = time.time()
    next_prediction_at = 0.0
    last_label  = ""
    last_tier   = ""

    try:
        while True:
            ok, frame_rgb = camera.read()
            if not ok or frame_rgb is None:
                print("[WARN] Failed to read frame.")
                time.sleep(0.05)
                continue

            frame_num += 1
            t_now  = time.time()
            fps    = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

            # ── KNN prediction every N frames ──────────────────────────────
            if (not frozen and frame_num % every == 0
                    and t_now >= next_prediction_at):
                feats = extract_features(extractor, frame_rgb)
                label, confidence, top_k = knn_predict(feats, db_features, db_labels, k=k)
                next_prediction_at = time.time() + RESULT_DISPLAY_SECONDS
                tier  = _confidence_tier(confidence)

                # Voice: announce on grade or tier change (same as live_predict.py)
                if voice_on and voice is not None:
                    if label != last_label or tier != last_tier:
                        voice.say(_build_speech_message(label, confidence))

                # Console verbose output
                if show_info and label != last_label:
                    print(f"\n[Predict] {label}  ({confidence*100:.0f}%)")
                    for lbl, sim in top_k:
                        print(f"  sim={sim:.3f}  {lbl}")

                last_label = label
                last_tier  = tier

                # OLED update — full rich layout
                _oled_live(oled_device, oled_driver, label, confidence, top_k, frozen)

            # ── GPIO button event handling ─────────────────────────────────
            if btn_mon is not None:
                ev = btn_mon.get_event()
                if ev in ("short", "double"):
                    # A single click must visibly save a frame in headless mode too.
                    fname = CAPTURES_DIR / f"{uuid.uuid4().hex[:8]}_{label.replace(' ','_')}.jpg"
                    _save_frame_rgb(frame_rgb, fname, cv2)
                    print(f"[BTN] Saved → {fname.name}")
                    if ev == "short" and label not in ("Initialising...", ""):
                        message = f"Frame saved. {_build_speech_message(label, confidence)}"
                    else:
                        message = "Frame saved. Prediction not ready yet." if label in (
                            "Initialising...", ""
                        ) else "Frame saved."
                    if voice_on and voice is not None:
                        voice.say(message)

                elif ev == "long":
                    # Freeze / unfreeze
                    frozen = not frozen
                    st = "Frozen" if frozen else "Resumed"
                    print(f"[BTN] {st}")
                    if voice_on and voice is not None:
                        voice.say(st)
                    _oled_live(oled_device, oled_driver, label, confidence, top_k, frozen)

            # ── OpenCV window & keyboard ───────────────────────────────────
            if cv2 is not None:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                display   = frame_bgr.copy()
                _draw_overlay(display, cv2, label, confidence, top_k,
                              fps, frozen, every, gpio_ok, voice_on)
                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord(" "):
                    frozen = not frozen
                    print(f"[Key] {'Frozen' if frozen else 'Resumed'}")
                    _oled_live(oled_device, oled_driver, label, confidence, top_k, frozen)
                elif key == ord("s"):
                    fname = CAPTURES_DIR / f"{uuid.uuid4().hex[:8]}_{label.replace(' ','_')}.jpg"
                    _save_frame_rgb(frame_rgb, fname, cv2)
                    print(f"[Key] Saved → {fname.name}")
                elif key == ord("f"):
                    show_info = not show_info
                    print(f"[Key] Console info {'ON' if show_info else 'OFF'}")
                elif key == ord("v"):
                    voice_on = not voice_on
                    st = "ON" if voice_on else "OFF"
                    print(f"[Key] Voice {st}")
                    if voice_on and voice is not None:
                        voice.say(f"Voice {st}")
                elif key == ord("a"):
                    # Manual announce
                    if label not in ("Initialising...", ""):
                        msg = _build_speech_message(label, confidence)
                        announce = f"Current prediction: {msg}"
                        print(f"[Key] {announce}")
                        if voice_on and voice is not None:
                            voice.say(announce)
                elif key == ord("+") or key == ord("="):
                    every = max(1, every - 5)
                    print(f"[Key] Predict every {every} frames")
                elif key == ord("-"):
                    every = min(60, every + 5)
                    print(f"[Key] Predict every {every} frames")
            else:
                # Headless: yield CPU
                time.sleep(0.033)

    except KeyboardInterrupt:
        print("\n[Aborted] Ctrl-C received.")
        if voice:
            voice.say("Shutting down.")

    finally:
        _oled_clear(oled_device, oled_driver)
        camera.release()
        if cv2 is not None:
            cv2.destroyAllWindows()
        if btn_mon is not None:
            btn_mon.stop()
        _cleanup_gpio()
        if voice is not None:
            voice.stop()
        print("[Cleanup] Done.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "MangoAI – Raspberry Pi LIVE mango grade predictor.\n\n"
            "Streams from Pi Camera (picamera2) or USB webcam and predicts the\n"
            "mango grade in real-time using KNN on CNN (MobileNetV3) features.\n"
            "Results are shown on the SSD1306 OLED, spoken via espeak-ng, and\n"
            "displayed in an OpenCV window with the same layout as live_predict.py.\n\n"
            "GPIO Button commands (GPIO 17, active-low pull-up):\n"
            "  Short press  (<1 s)       : Save current frame and announce prediction\n"
            "  Long  press  (>=1 s)      : Freeze / unfreeze prediction updates\n"
            "  Double press (<=0.6 s gap): Save current frame to disk\n\n"
            "Keyboard shortcuts (OpenCV window):\n"
            "  Q / ESC  Quit          SPACE  Freeze/unfreeze\n"
            "  S        Save frame    A      Announce grade\n"
            "  V        Toggle voice  F      Toggle console info\n"
            "  + / =    Speed up      -      Slow down"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, metavar="N",
        help="USB camera index (default 0). Ignored if Pi Camera found.",
    )
    parser.add_argument(
        "--k", type=int, default=5, metavar="K",
        help="K nearest neighbours for voting (default 5).",
    )
    parser.add_argument(
        "--every", type=int, default=15, metavar="N",
        help="Predict every N frames (default 15).",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Headless mode: skip OpenCV preview window.",
    )
    parser.add_argument(
        "--no-oled", action="store_true",
        help="Skip SSD1306 OLED output.",
    )
    parser.add_argument(
        "--no-button", action="store_true",
        help="Skip GPIO button (keyboard-only).",
    )
    parser.add_argument(
        "--no-voice", action="store_true",
        help="Disable voice announcements.",
    )
    args = parser.parse_args()

    run(
        camera_index=args.camera_index,
        k=args.k,
        every=args.every,
        use_display=not args.no_display,
        use_oled=not args.no_oled,
        use_button=not args.no_button,
        use_voice=not args.no_voice,
    )
