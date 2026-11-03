# Mango grading project

This project grades mangoes from four camera views using a trained TensorFlow model and a majority-vote workflow. It supports a desktop camera workflow, an upload-based web app, and a Raspberry Pi capture flow with an optional OLED display.

## What this project does

- Captures four images in the order: Right, Left, Front, Back
- Runs prediction on each view individually
- Combines the four predictions into a single result using majority voting
- Supports optional validity detection for mango/non-mango checks
- Provides a local Flask web app for uploading images
- Includes a Raspberry Pi workflow for a Pi camera or USB webcam

## Project layout

- `use_model.py` — model loading, preprocessing, inference, and four-view aggregation
- `web_app.py` — Flask upload app for grading uploaded images
- `laptop_camera_test.py` — local laptop camera capture test script
- `live_predict.py` — live prediction flow for desktop camera input
- `rpi_camera.py` — Raspberry Pi camera workflow with OLED output
- `rpi_live_predict.py` — Raspberry Pi live prediction variant
- `shape_features.py` — shape-feature extraction used by some model inputs
- `shape_gate.py` — shape-based gate logic
- `tests/` — unit tests for workflow and web app validation
- `captures/` — saved session outputs from camera runs
- `mobilenetv3_runs/` — default project grade model folder

## Requirements

Python dependencies are listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The project also expects a working TensorFlow installation and model files in the repo or a configured sibling folder.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Make sure the trained model artifact is available.

The project looks for the grade model in these locations, in order:

- `mobilenetv3_runs/best_model.keras`
- `mobilenetv3_runs/final_model.keras`
- `final_model.keras`
- optionally a sibling `../pro copy 2/mobilenetv3_runs/...` location

The class names must be available next to the model, typically as `class_names.json` or `mobilenetv3_runs/class_names.json`.

If a validity model is used, it should be present at one of:

- `validity_runs/best_validity.keras`
- `validity_runs/final_validity.keras`
- `final_validity.keras`

and its class map at one of:

- `validity_runs/class_names.json`
- `validity_class_names.json`

## Desktop camera workflow

From the project root:

```bash
python use_model.py --camera
```

This launches the desktop camera capture flow. The user can click the preview to save each side in the order:

- Right
- Left
- Front
- Back

The session output is saved under `captures/<session_id>/` with raw images and processed arrays.

## Upload web app

Start the local upload site:

```bash
python web_app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Upload one image for each side (Front, Back, Left, Right), then click **Detect grade**. The app converts each upload to RGB PNG, predicts each view, and combines the four results.

## Raspberry Pi setup

For the Pi-specific workflow, use the Raspberry Pi dependencies:

```bash
pip install -r requirements_rpi.txt
```

Then run:

```bash
python rpi_camera.py
```

This can:

- auto-detect a Pi camera or USB webcam
- capture four views using a button or keyboard Enter
- show the final grade on an SSD1306 OLED display

See `CAMERA_WORKFLOW.md` for hardware wiring details, OLED setup, and capture flow.

## Running tests

```bash
python -m unittest discover -s tests -v
```

## Important notes

- This project expects trained model artifacts to exist before prediction can run.
- If a model or class mapping is missing, the app or CLI will warn and fail gracefully instead of crashing.
- The four-view workflow produces a final grade from combined side predictions.
- The project includes optional speech output, depending on the host OS:
  - macOS: `say`
  - Linux/RPi: `espeak-ng` or `espeak`

## Related docs

- `CAMERA_WORKFLOW.md` — detailed camera and upload workflow documentation
- `requirements.txt` — desktop dependencies
- `requirements_rpi.txt` — Raspberry Pi dependencies

## License

This project does not currently include a license file. Add one if you plan to distribute or publish the code.
