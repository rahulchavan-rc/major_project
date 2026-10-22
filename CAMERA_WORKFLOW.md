# Four-view camera workflow

Run from the project directory:

```sh
python -m pip install -r requirements.txt
python use_model.py --camera
```

Run software checks with:

```sh
python -m unittest discover -s tests -v
```

## Image upload website

Install the dependencies above, then start the local upload website with:

```sh
python web_app.py
```

Open `http://127.0.0.1:5000`, choose a JPEG, PNG, WebP, BMP, or TIFF image, and
select **Detect grade**. The website converts uploads to RGB PNG before using
the same preprocessing and trained model as image-file prediction. Uploads are
limited to 50 MB and 20 megapixels and are removed after prediction. The trained
grade model and its class mapping must be available at one of the model paths
described below.

Use `--camera-index N` to select another OpenCV camera. The live preview remains
active while capturing. Click it once to save each side, in this order:
**Right -> Left -> Front -> Back**. The next side becomes active only after its
image has been written successfully. Any visible object can be captured.

Each run creates an isolated session:

```text
captures/<session_id>/
  raw/right.jpg
  raw/left.jpg
  raw/front.jpg
  raw/back.jpg
  processed/right.npy
  processed/left.npy
  processed/front.npy
  processed/back.npy
  results.json
```

Captured images are not read from the training dataset. During the result screen,
press `1`, `2`, `3`, or `4` to select Right, Left, Front, or Back for a retake,
then click once to replace that side. A successful retake reruns the four-view
prediction. Press `q` to close; any successfully saved files remain in that
session. There is no countdown, automatic capture, or motor-control code in
this project.

## Models and validity limitations

The camera can capture and save images even when models are missing. Prediction
searches for a trained grade model in the current project first:

- `mobilenetv3_runs/best_model.keras`
- `mobilenetv3_runs/final_model.keras`
- `final_model.keras`

For this Desktop setup, it also checks the sibling
`../pro copy 2/mobilenetv3_runs/` folder after the current project's local
MobileNetV3 files. The class mapping is loaded from beside the selected model,
so class indices remain paired with that artifact. The camera command loads
the selected model before opening the preview; all four saved sides are then
predicted through the same preprocessing and model path as image-file
prediction, and combined into one object result.

For file/image prediction, if none is present the command prints
`PREDICTION UNAVAILABLE` and lists the expected paths instead of returning a
Python traceback. No grade can be produced until the existing trained model
artifact is restored in one of those locations.

The grade class order comes from `mobilenetv3_runs/class_names.json` or
`class_names.json`. A validity detector is optional; to use it, provide one of
the configured validity `.keras` files and a companion binary mapping at
`validity_runs/class_names.json` or `validity_class_names.json`, with explicit
Mango/Non-Mango class labels. If no usable detector and mapping exist, the UI
labels the grade output as unvalidated and says grades are for mangoes only.
An explicit Non-Mango detector prediction produces “Not a mango—grading
unavailable.”

The current inference helper deterministically decodes RGB, stretches each
image to 224×224 with TensorFlow bilinear resizing, then casts to `float32` in
the 0–255 range. The two-input grade model path also computes its existing
shape-feature input from that resized image. These are facts about the
inference code only: this project copy contains no training scripts or saved
grade, validity, or shape-gate model files, so compatibility with training
preprocessing and any preprocessing embedded in those models cannot be
verified. The result metadata records this as unverified. Supply the training
source and matching model artifacts before treating predictions as validated.

Four external RGB views cannot establish a mango's internal condition. No
accuracy improvement is claimed; it needs evaluation on appropriate held-out
data.

## Offline speech and hardware checks

- macOS uses its built-in `say` command.
- Linux/Raspberry Pi uses `espeak-ng` or `espeak` when available. On Debian or
  Raspberry Pi OS, install it with `sudo apt install espeak-ng`.
- Speech runs on a background worker, one confirmation at a time. If no speech
  command is installed, visible prompts and results remain available.

Software checks can exercise ordering, atomic saves, session isolation,
retakes, preprocessing, and vote/tie handling. A real camera is still required
to verify camera permissions, preview, focus, and framing. Audible confirmation
requires the relevant local speech command and audio output.

---

## Raspberry Pi camera + OLED display (`rpi_camera.py`)

`rpi_camera.py` is the dedicated Raspberry Pi runner. It auto-detects the
camera (Pi Camera Module **or** USB webcam), captures four views guided by a
physical button or keyboard Enter, runs the 4-view grade prediction, and shows
the final result on a **1.3-inch SSD1306 OLED display** (128×64, I2C).

### Hardware wiring

```
OLED (SSD1306 – I2C)
  GND  →  Pin 6   (GND)
  VCC  →  Pin 1   (3.3 V)
  SCL  →  Pin 5   (GPIO 3 / SCL1)
  SDA  →  Pin 3   (GPIO 2 / SDA1)

Capture button (optional, active-low)
  One leg  →  Pin 11  (GPIO 17)
  Other    →  Pin 9   (GND)   ← internal pull-up is enabled automatically
```

### Raspberry Pi OS setup

**1. Enable I2C** (needed for the OLED):

```sh
sudo raspi-config
# Interface Options → I2C → Enable
sudo reboot
```

Verify the display is seen at address `0x3C`:

```sh
sudo apt install -y i2c-tools
i2cdetect -y 1
```

**2. Pi Camera Module** (skip if using a USB webcam):

```sh
sudo apt install -y python3-picamera2 python3-libcamera
# Edit /boot/config.txt and confirm: camera_auto_detect=1
sudo reboot
# Quick test:
libcamera-hello --timeout 2000
```

**3. Install Python dependencies**:

```sh
pip install -r requirements_rpi.txt
# luma.oled pulls in Pillow and smbus2 automatically.
```

> **Note**: `tensorflow` may need the Raspberry Pi–specific wheel on arm64.
> If `pip install tensorflow` fails, try:
> ```sh
> pip install tensorflow-aarch64
> ```

### Running the grader

```sh
# Auto-detect camera + show result on OLED:
python rpi_camera.py

# Force USB camera index 1:
python rpi_camera.py --camera-index 1

# Terminal output only (no OLED):
python rpi_camera.py --no-display

# Keyboard Enter only (no GPIO button):
python rpi_camera.py --no-button
```

### Capture flow

1. Script loads the trained grade model.
2. OLED shows **"Show Right — Press button"**.
3. Position the mango, press the button (or Enter).
4. Repeat for **Left → Front → Back**.
5. Model runs 4-view inference (majority-vote).
6. OLED shows the **final grade** and confidence score:

```
  MangoAI Result
────────────────
  Grade 1
  94.3%  4/4 votes
```

If the object is not a mango or the validity model rejects it:

```
  MangoAI
────────────────
  INVALID
  Not a mango
```

### Session output

Each run creates an isolated session folder identical to the desktop workflow:

```text
captures/<session_id>/
  raw/right.jpg
  raw/left.jpg
  raw/front.jpg
  raw/back.jpg
  results.json
```

### Camera auto-detection order

| Priority | Driver | Requirement |
|----------|--------|-------------|
| 1st | `picamera2` (Pi Camera) | `python3-picamera2` installed |
| 2nd | OpenCV (`cv2`) | `opencv-python` installed |

If neither is available the script exits with a clear install message.

### OLED driver auto-detection order

| Priority | Library | Install |
|----------|---------|---------|
| 1st | `luma.oled` | `pip install luma.oled` |
| 2nd | `Adafruit_SSD1306` | `pip install Adafruit-SSD1306` |

If no OLED driver is found the result is printed to the terminal only.

