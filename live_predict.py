"""
live_predict.py – MangoAI webcam predictor (live + 4-side capture modes).

MODES
──────
  Live mode (default):
    Continuously predicts grade from the webcam feed in real-time using
    K-nearest-neighbour matching against saved training features.

  Capture mode (--capture):
    Guides you to capture 4 sides (Right → Left → Front → Back), then
    runs KNN on each side, majority-votes the final grade, and announces
    the result with voice.

LIVE MODE CONTROLS
  Q         – Quit
  S         – Save current frame
  SPACE     – Freeze / unfreeze
  + / -     – Speed up / slow down predictions
  F         – Toggle console info
  V         – Toggle voice

CAPTURE MODE CONTROLS
  SPACE / click – Capture current side
  1 2 3 4       – Retake Right / Left / Front / Back
  Q             – Quit
  V             – Toggle voice

Usage:
    python live_predict.py                  # live mode
    python live_predict.py --capture        # 4-side capture mode
    python live_predict.py --capture --k 7
    python live_predict.py --no-voice
"""

from __future__ import annotations

import argparse
import json
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from collections import Counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PATH   = PROJECT_ROOT / "extracted_features.json"
CAPTURES_DIR    = PROJECT_ROOT / "captures" / "live"

# Grade label mapping (valid_mango label contains grade via image path)
GRADE_KEYWORDS = {
    "grade 1": "Grade 1",
    "grade 2": "Grade 2",
    "grade 3": "Grade 3",
    "grade 4": "Grade 4",
}

# Confidence tiers for voice announcements
CONF_TIERS = [
    (0.80, "very high"),
    (0.60, "high"),
    (0.40, "medium"),
    (0.00, "low"),
]


# ── Self-contained speech helper ─────────────────────────────────────────────

class _Speech:
    """Non-blocking text-to-speech using macOS 'say' or Linux espeak-ng."""

    def __init__(self) -> None:
        system = platform.system()
        if system == "Darwin" and shutil.which("say"):
            self.cmd = "say"
        elif system == "Linux":
            self.cmd = shutil.which("espeak-ng") or shutil.which("espeak")
        else:
            self.cmd = None

        self._q: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()

        if self.cmd is None:
            print("[Voice] No speech command found (macOS 'say' or Linux 'espeak-ng'). "
                  "Voice announcements disabled.")
        else:
            print(f"[Voice] Using '{self.cmd}' for announcements.")
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def say(self, text: str) -> None:
        if self.cmd and not self._stop.is_set():
            # Drain queue so only the latest message plays
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._q.put(text)

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

    def stop(self) -> None:
        self._stop.set()
        self._q.put(None)
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()


def _confidence_tier(confidence: float) -> str:
    """Return a human-readable confidence tier string."""
    for threshold, label in CONF_TIERS:
        if confidence >= threshold:
            return label
    return "low"


def _build_speech_message(label: str, confidence: float) -> str:
    """Build a natural-sounding voice announcement."""
    pct  = int(round(confidence * 100))
    tier = _confidence_tier(confidence)
    if label == "invalid":
        return f"Invalid. Not a mango. Confidence {pct} percent."
    return f"{label}. {tier} confidence. {pct} percent."

# ── Colours (BGR for OpenCV) ───────────────────────────────────────────────────
CLR_GREEN  = (50, 220, 80)
CLR_RED    = (60, 60, 240)
CLR_YELLOW = (0, 200, 255)
CLR_WHITE  = (255, 255, 255)
CLR_BLACK  = (0, 0, 0)
CLR_DARK   = (20, 20, 20)
CLR_NEON   = (100, 255, 100)


# ══════════════════════════════════════════════════════════════════════════════
# Feature database
# ══════════════════════════════════════════════════════════════════════════════

def _parse_grade_from_path(image_path: str) -> str:
    """Extract grade from the training image path."""
    lower = image_path.lower()
    for kw, label in GRADE_KEYWORDS.items():
        if kw in lower:
            return label
    return "Unknown"


def load_feature_db(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load extracted_features.json.
    Returns:
        features   – (N, D) float32 array, L2-normalised
        labels     – list of 'Grade 1' … 'Grade 4' or 'invalid'
        img_paths  – list of original image paths
    """
    print(f"[DB] Loading feature database from {path.name}…")
    with path.open("r", encoding="utf-8") as f:
        db = json.load(f)

    samples   = db["samples"]
    raw_feats = np.array([s["features"] for s in samples], dtype=np.float32)

    # L2-normalise each row for fast cosine similarity via dot product
    norms = np.linalg.norm(raw_feats, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    features = raw_feats / norms

    labels = []
    for s in samples:
        if s["label"] == "invalid":
            labels.append("invalid")
        else:
            labels.append(_parse_grade_from_path(s["image_path"]))

    img_paths = [s["image_path"] for s in samples]

    grade_counts = Counter(l for l in labels if l != "invalid")
    print(f"[DB] {len(samples)} samples | valid: {sum(1 for l in labels if l != 'invalid')} | invalid: {sum(1 for l in labels if l == 'invalid')}")
    print(f"[DB] Grade distribution: {dict(grade_counts)}")
    return features, labels, img_paths


# ══════════════════════════════════════════════════════════════════════════════
# Model feature extractor (same backbone as training)
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_extractor():
    """
    Build a 576-dim CNN feature extractor.
    Priority:
      1. Tap 'cnn_features' layer of the trained project model
         (avoids ImageNet weight download and macOS malloc crashes).
      2. Fresh MobileNetV3Small with ImageNet weights as fallback.
    """
    import tensorflow as tf
    from pathlib import Path

    # ── Try loading the existing trained grade model ──────────────────────
    GRADE_CANDIDATES = [
        PROJECT_ROOT / "mobilenetv3_runs" / "best_model.keras",
        PROJECT_ROOT / "mobilenetv3_runs" / "final_model.keras",
        PROJECT_ROOT.parent / "pro copy 2" / "mobilenetv3_runs" / "best_model.keras",
        PROJECT_ROOT.parent / "pro copy 2" / "mobilenetv3_runs" / "final_model.keras",
    ]
    grade_path = next((p for p in GRADE_CANDIDATES if p.exists()), None)

    if grade_path:
        print(f"[Model] Loading trained model for feature extraction: {grade_path.name}")
        try:
            full_model = tf.keras.models.load_model(
                grade_path,
                custom_objects={
                    "GlorotUniform": tf.keras.initializers.GlorotUniform,
                    "GlorotNormal":  tf.keras.initializers.GlorotNormal,
                },
                compile=False,
            )
            # Find the CNN features layer (global average pool output)
            feat_layer = None
            for layer in reversed(full_model.layers):
                shape = getattr(layer.output_shape, '__iter__', None)
                try:
                    out_shape = layer.output_shape
                    if isinstance(out_shape, (list, tuple)) and len(out_shape) == 2:
                        dim = out_shape[-1]
                        if isinstance(dim, int) and 256 <= dim <= 1024:
                            feat_layer = layer
                            break
                except Exception:
                    continue

            if feat_layer is not None:
                extractor = tf.keras.Model(
                    inputs=full_model.input,
                    outputs=feat_layer.output,
                )
                print(f"[Model] Feature extractor → layer '{feat_layer.name}' "
                      f"shape {feat_layer.output_shape}")
                return extractor
            else:
                # Use penultimate layer
                extractor = tf.keras.Model(
                    inputs=full_model.inputs,
                    outputs=full_model.layers[-2].output,
                )
                print(f"[Model] Feature extractor → penultimate layer "
                      f"{full_model.layers[-2].name}")
                return extractor
        except Exception as exc:
            print(f"[Model] Could not tap trained model ({exc}), falling back.")

    # ── Fallback: fresh MobileNetV3Small ─────────────────────────────────
    print("[Model] Building fresh MobileNetV3-Small feature extractor…")
    from tensorflow.keras.applications import MobileNetV3Small
    base = MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False
    extractor = tf.keras.Model(inputs=base.input, outputs=base.output)
    print(f"[Model] Feature extractor ready — output shape: {extractor.output_shape}")
    return extractor


def extract_features(model, frame_rgb: np.ndarray) -> np.ndarray:
    """
    Preprocess an RGB frame to 224x224 float32 and extract CNN features.
    Uses direct model call (not model.predict) to avoid macOS malloc crashes.
    Returns a L2-normalised 1-D numpy array.
    """
    import tensorflow as tf
    img = tf.image.resize(
        tf.constant(frame_rgb, dtype=tf.float32),
        [224, 224],
        method="bilinear",
    )
    img = tf.expand_dims(img, 0)          # (1, 224, 224, 3)
    # Direct call instead of model.predict() — avoids macOS TF malloc crash
    feats = model(img, training=False).numpy()[0]
    norm = np.linalg.norm(feats)
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
        predicted_label  – e.g. 'Grade 2' or 'invalid'
        confidence       – fraction of top-K that voted for winner
        top_k            – list of (label, similarity) for the K neighbours
    """
    sims   = db_features @ query          # dot product == cosine sim (L2-normed)
    top_k_idx = np.argpartition(sims, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]

    top_k     = [(db_labels[i], float(sims[i])) for i in top_k_idx]
    votes     = Counter(label for label, _ in top_k)
    winner    = votes.most_common(1)[0][0]
    confidence = votes[winner] / k

    return winner, confidence, top_k


# ══════════════════════════════════════════════════════════════════════════════
# Overlay rendering
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
) -> None:
    h, w = frame.shape[:2]

    # ── Top bar ────────────────────────────────────────────────────────────
    bar_h = 62
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), CLR_DARK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    is_invalid = label == "invalid"
    grade_colour = CLR_RED if is_invalid else CLR_NEON
    grade_text   = "INVALID – Not a mango" if is_invalid else label.upper()

    cv2.putText(frame, grade_text,
                (14, 38), cv2.FONT_HERSHEY_DUPLEX, 1.1,
                grade_colour, 2, cv2.LINE_AA)

    conf_text = f"{confidence*100:.0f}% confidence"
    cv2.putText(frame, conf_text,
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                CLR_WHITE, 1, cv2.LINE_AA)

    # ── Vote bars (right side) ─────────────────────────────────────────────
    vote_labels = [lbl for lbl, _ in top_k]
    vote_counts = Counter(vote_labels)
    all_grades  = ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "invalid"]
    bar_x  = w - 160
    bar_y0 = 80
    bw     = 130
    bh     = 14

    # background
    cv2.rectangle(frame, (bar_x - 8, bar_y0 - 20), (w - 4, bar_y0 + len(all_grades) * 24 + 4), CLR_DARK, -1)

    for i, g in enumerate(all_grades):
        cnt  = vote_counts.get(g, 0)
        frac = cnt / max(sum(vote_counts.values()), 1)
        y    = bar_y0 + i * 24
        col  = CLR_NEON if g == label else (100, 100, 100)

        # label
        short = g.replace("Grade ", "G") if "Grade" in g else "INV"
        cv2.putText(frame, short, (bar_x - 8, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        # track
        cv2.rectangle(frame, (bar_x + 22, y), (bar_x + 22 + bw, y + bh), (50, 50, 50), -1)
        # fill
        if frac > 0:
            cv2.rectangle(frame, (bar_x + 22, y),
                          (bar_x + 22 + int(bw * frac), y + bh), col, -1)
        # count
        cv2.putText(frame, f"{cnt}", (bar_x + 22 + bw + 4, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    # ── Bottom status bar ──────────────────────────────────────────────────
    status = []
    if frozen:
        status.append("FROZEN")
    status.append(f"FPS:{fps:.1f}")
    status.append(f"every:{every}fr")
    status.append("Q=quit S=save SPACE=freeze +/-=speed F=info")
    cv2.putText(frame, "  ".join(status),
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                CLR_YELLOW, 1, cv2.LINE_AA)

    # ── MangoAI watermark ─────────────────────────────────────────────────
    cv2.putText(frame, "MangoAI",
                (w - 80, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (60, 60, 60), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# Main live loop
# ══════════════════════════════════════════════════════════════════════════════

def run(camera_index: int = 0, k: int = 5, every: int = 15, voice: bool = True) -> None:
    try:
        import cv2
    except ImportError:
        raise SystemExit("Install opencv-python: pip install opencv-python")

    # Load feature DB
    if not FEATURES_PATH.exists():
        raise SystemExit(f"Feature database not found: {FEATURES_PATH}")
    db_features, db_labels, db_paths = load_feature_db(FEATURES_PATH)

    # Build feature extractor
    extractor = build_feature_extractor()

    # Open camera
    print(f"[Camera] Opening camera index {camera_index}…")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {camera_index}.")

    # Warm up
    for _ in range(5):
        cap.read()

    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Voice / speech ────────────────────────────────────────────────────
    speech = _Speech() if voice else None

    print("\n" + "=" * 60)
    print("  MangoAI – Live Camera Predictor")
    print("=" * 60)
    print(f"  K-nearest  : {k}")
    print(f"  Predict every {every} frames")
    print(f"  Voice      : {'ON' if voice else 'OFF'}")
    print(f"  Controls   : Q=quit  S=save  SPACE=freeze  +/-=speed  F=info  V=voice")
    print("=" * 60 + "\n")

    label      = "Initialising…"
    confidence = 0.0
    top_k      = []
    frozen     = False
    show_info  = False
    voice_on   = voice        # can be toggled live with V key
    frame_num  = 0
    fps        = 0.0
    t_prev     = time.time()
    last_label = ""
    last_tier  = ""           # track confidence tier changes

    window_name = "MangoAI – Live Prediction  (Q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 500)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame.")
                continue

            frame_num += 1
            t_now = time.time()
            fps   = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

            # Predict every N frames (if not frozen)
            if not frozen and frame_num % every == 0:
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feats = extract_features(extractor, rgb)
                label, confidence, top_k = knn_predict(feats, db_features, db_labels, k=k)
                tier = _confidence_tier(confidence)

                # ── Voice: announce on grade change OR confidence tier change ──
                if voice_on and speech is not None:
                    if label != last_label or tier != last_tier:
                        msg = _build_speech_message(label, confidence)
                        speech.say(msg)
                        print(f"[Voice] {msg}")

                if show_info and label != last_label:
                    print(f"\n[Predict] {label}  ({confidence*100:.0f}%)")
                    for lbl, sim in top_k:
                        print(f"  sim={sim:.3f}  {lbl}")

                last_label = label
                last_tier  = tier

            # Draw overlay
            display = frame.copy()
            _draw_overlay(display, cv2, label, confidence, top_k, fps, frozen, every)
            cv2.imshow(window_name, display)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                frozen = not frozen
                print(f"[Info] {'Frozen' if frozen else 'Resumed'}")
            elif key == ord("s"):
                fname = CAPTURES_DIR / f"{uuid.uuid4().hex[:8]}_{label.replace(' ','_')}.jpg"
                cv2.imwrite(str(fname), frame)
                print(f"[Save] {fname.name}")
            elif key == ord("f"):
                show_info = not show_info
                print(f"[Info] Console info {'ON' if show_info else 'OFF'}")
            elif key == ord("v"):
                voice_on = not voice_on
                status = "ON" if voice_on else "OFF"
                print(f"[Voice] Toggled {status}")
                if voice_on and speech is not None:
                    speech.say(f"Voice {status}")
            elif key == ord("+") or key == ord("="):
                every = max(1, every - 5)
                print(f"[Info] Predict every {every} frames")
            elif key == ord("-"):
                every = min(60, every + 5)
                print(f"[Info] Predict every {every} frames")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if speech is not None:
            speech.stop()
        print("[Done] Camera closed.")


CAMERA_SIDES = ("right", "left", "front", "back")
SIDE_ICONS   = {"right": "▶", "left": "◀", "front": "▲", "back": "▼"}


# ══════════════════════════════════════════════════════════════════════════════
# 4-side capture + predict mode
# ══════════════════════════════════════════════════════════════════════════════

def _draw_capture_overlay(
    frame, cv2,
    state: str,
    current_side: str,
    captured: dict,
    result: dict | None,
    notice: str,
) -> None:
    """Draw HUD overlay on the frame for capture mode."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    bar_h = 100
    cv2.rectangle(overlay, (0, 0), (w, bar_h), CLR_DARK, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    if state == "result" and result:
        grade = result.get("final_grade", "")
        conf  = result.get("confidence", 0) * 100
        votes = result.get("winning_votes", "?")
        is_invalid = not grade or grade == "invalid"
        col   = CLR_RED if is_invalid else CLR_NEON
        text  = "INVALID – Not a mango" if is_invalid else grade.upper()
        cv2.putText(frame, "RESULT:", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, text, (12, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.15, col, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{conf:.0f}% confidence  |  {votes}/4 sides agreed",
                    (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "1-4=Retake  Q=Quit  V=Voice",
                    (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_YELLOW, 1, cv2.LINE_AA)
    else:
        icon = SIDE_ICONS.get(current_side, "")
        cv2.putText(frame, f"Capture: {icon} {current_side.upper()} side",
                    (12, 34), cv2.FONT_HERSHEY_DUPLEX, 0.9, CLR_NEON, 2, cv2.LINE_AA)
        cv2.putText(frame, "Press SPACE to capture",
                    (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_WHITE, 1, cv2.LINE_AA)
        if notice:
            cv2.putText(frame, notice, (12, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, "Q=Quit  V=Voice",
                    (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_YELLOW, 1, cv2.LINE_AA)

    # Side progress strip at bottom-right
    strip_x = w - 160
    strip_y = h - 80
    for i, side in enumerate(CAMERA_SIDES):
        done = side in captured
        col  = CLR_NEON if done else (80, 80, 80)
        mark = "✓" if done else SIDE_ICONS[side]
        label = f"{mark} {side.title()}"
        cv2.putText(frame, label, (strip_x, strip_y + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)


def _knn_predict_frame(
    frame_bgr,
    cv2,
    extractor,
    db_features: np.ndarray,
    db_labels: list[str],
    k: int,
) -> tuple[str, float, list]:
    """Run KNN prediction on a single BGR frame."""
    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    feats = extract_features(extractor, rgb)
    return knn_predict(feats, db_features, db_labels, k=k)


def _combine_side_results(
    side_results: dict[str, tuple[str, float, list]],
) -> dict:
    """Majority-vote across 4 side predictions."""
    labels = [side_results[s][0] for s in CAMERA_SIDES]
    votes  = Counter(labels)

    # If any side says invalid, result is invalid
    if "invalid" in votes and votes["invalid"] >= 1:
        return {
            "final_grade":    "invalid",
            "confidence":      0.0,
            "winning_votes":   votes["invalid"],
            "vote_counts":     dict(votes),
            "per_side":        {s: side_results[s][0] for s in CAMERA_SIDES},
        }

    winner = votes.most_common(1)[0][0]
    # Average confidence across sides that voted for winner
    winner_confs = [
        side_results[s][1]
        for s in CAMERA_SIDES
        if side_results[s][0] == winner
    ]
    avg_conf = float(np.mean(winner_confs))
    return {
        "final_grade":    winner,
        "confidence":      avg_conf,
        "winning_votes":   votes[winner],
        "vote_counts":     dict(votes),
        "per_side":        {s: side_results[s][0] for s in CAMERA_SIDES},
    }


def run_capture_mode(
    camera_index: int = 0,
    k: int = 5,
    voice: bool = True,
) -> None:
    """4-side capture → KNN predict → voice announce."""
    try:
        import cv2
    except ImportError:
        raise SystemExit("Install opencv-python: pip install opencv-python")

    if not FEATURES_PATH.exists():
        raise SystemExit(f"Feature database not found: {FEATURES_PATH}")

    db_features, db_labels, _ = load_feature_db(FEATURES_PATH)
    extractor = build_feature_extractor()
    speech    = _Speech() if voice else None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {camera_index}.")
    for _ in range(5):
        cap.read()

    session_id  = uuid.uuid4().hex[:10]
    session_dir = PROJECT_ROOT / "captures" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  MangoAI – 4-Side Capture Mode")
    print("=" * 60)
    print(f"  Session : {session_id}")
    print(f"  Order   : Right → Left → Front → Back")
    print(f"  Voice   : {'ON' if voice else 'OFF'}")
    print(f"  Controls: SPACE=capture  1-4=retake  Q=quit  V=voice")
    print("=" * 60 + "\n")

    if speech:
        speech.say("Capture mode ready. Show the right side and press space.")

    captured:     dict[str, np.ndarray] = {}   # side → raw BGR frame
    side_results: dict[str, tuple]      = {}   # side → (label, conf, top_k)
    side_index = 0
    state      = "capturing"   # 'capturing' | 'predicting' | 'result'
    result     = None
    notice     = ""
    window     = "MangoAI – 4-Side Capture  (SPACE=capture  Q=quit)"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 800, 520)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            current_side = CAMERA_SIDES[min(side_index, len(CAMERA_SIDES) - 1)]
            display = frame.copy()
            _draw_capture_overlay(display, cv2, state, current_side,
                                  captured, result, notice)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF

            # ── Quit ──────────────────────────────────────────────────────
            if key == ord("q") or key == 27:
                break

            # ── Voice toggle ──────────────────────────────────────────────
            elif key == ord("v"):
                if speech:
                    speech.say("Voice off")
                    speech = None
                else:
                    speech = _Speech()
                    speech.say("Voice on")

            # ── Retake (1-4 keys) on result screen ────────────────────────
            elif state == "result" and ord("1") <= key <= ord("4"):
                side_index = key - ord("1")
                retake_side = CAMERA_SIDES[side_index]
                captured.pop(retake_side, None)
                side_results.pop(retake_side, None)
                state  = "capturing"
                result = None
                notice = f"Retaking {retake_side.title()} side…"
                if speech:
                    speech.say(f"Retaking {retake_side} side.")
                print(f"[Retake] {retake_side.title()} side")

            # ── Capture SPACE ─────────────────────────────────────────────
            elif key == ord(" ") and state == "capturing":
                side = CAMERA_SIDES[side_index]
                captured[side] = frame.copy()

                # Save frame
                img_path = session_dir / f"{side}.jpg"
                cv2.imwrite(str(img_path), frame)
                print(f"[Capture] {side.title()} saved → {img_path.name}")

                # KNN predict this side
                notice = f"Predicting {side.title()}…"
                _draw_capture_overlay(display, cv2, "predicting", side,
                                      captured, None, notice)
                cv2.imshow(window, display)
                cv2.waitKey(1)

                lbl, conf, topk = _knn_predict_frame(
                    frame, cv2, extractor, db_features, db_labels, k
                )
                side_results[side] = (lbl, conf, topk)
                print(f"[Predict] {side.title()}: {lbl}  ({conf*100:.0f}%)")
                if speech:
                    speech.say(f"{side} side: {lbl}.")

                # Advance to next side
                side_index += 1
                notice = f"{side.title()} ✓ — {lbl} ({conf*100:.0f}%)"

                # All 4 sides captured → combine
                if len(captured) == 4:
                    state  = "predicting"
                    notice = "Combining predictions…"
                    result = _combine_side_results(side_results)
                    state  = "result"

                    grade    = result["final_grade"]
                    conf_avg = result["confidence"]
                    w_votes  = result["winning_votes"]
                    tier     = _confidence_tier(conf_avg)
                    per_side = result["per_side"]

                    print("\n" + "=" * 60)
                    print("  FINAL RESULT")
                    print("=" * 60)
                    for s in CAMERA_SIDES:
                        mark = "✓" if per_side[s] != "invalid" else "✗"
                        print(f"  {mark} {s.title():6s}: {per_side[s]}")
                    print(f"  {'─'*30}")
                    if grade == "invalid":
                        print("  Grade  : INVALID – Not a mango")
                    else:
                        print(f"  Grade  : {grade.title()}")
                        print(f"  Conf   : {conf_avg*100:.1f}%  ({tier})")
                        print(f"  Votes  : {w_votes}/4 sides agreed")
                    print("=" * 60 + "\n")

                    # Voice announcement
                    if speech:
                        msg = _build_speech_message(grade, conf_avg)
                        speech.say(f"Final result. {msg}")
                        print(f"[Voice] Final result. {msg}")

                    notice = f"Press 1-4 to retake a side, Q to quit"

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if speech:
            speech.stop()
        print("[Done] Session saved to:", session_dir)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MangoAI – webcam mango grader (live or 4-side capture).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--capture", action="store_true",
                        help="4-side capture mode (Right→Left→Front→Back then predict).")
    parser.add_argument("--camera-index", type=int, default=0, metavar="N",
                        help="Camera device index (default: 0)")
    parser.add_argument("--k", type=int, default=5, metavar="K",
                        help="K nearest neighbours for voting (default: 5)")
    parser.add_argument("--every", type=int, default=15, metavar="N",
                        help="[Live mode] Predict every N frames (default: 15)")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable voice announcements.")
    args = parser.parse_args()

    if args.capture:
        run_capture_mode(
            camera_index=args.camera_index,
            k=args.k,
            voice=not args.no_voice,
        )
    else:
        run(
            camera_index=args.camera_index,
            k=args.k,
            every=args.every,
            voice=not args.no_voice,
        )
