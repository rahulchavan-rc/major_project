from __future__ import annotations

import logging
import tempfile
import threading
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

import use_model


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_model_lock = threading.Lock()
_loaded_models: tuple[object | None, object, list[str], list[str] | None] | None = None


def _load_models_now():
    """Load all models and cache them. Safe to call from any thread."""
    global _loaded_models
    if _loaded_models is None:
        with _model_lock:
            if _loaded_models is None:
                grade_model    = use_model.load_grade_model()
                class_names    = use_model.load_class_names()
                validity_model = use_model.load_validity_model()
                validity_names = (
                    use_model.load_validity_class_names()
                    if validity_model is not None
                    else None
                )
                _loaded_models = (
                    validity_model,
                    grade_model,
                    class_names,
                    validity_names,
                )


def _get_models() -> tuple[object | None, object, list[str], list[str] | None]:
    _load_models_now()
    return _loaded_models


# ── Pre-load models in the background so the first HTTP request is instant ───
_preload_thread = threading.Thread(target=_load_models_now, daemon=True, name="model-preload")
_preload_thread.start()


def _render(message: str | None = None, result: dict | None = None, status: int = 200):
    return render_template("index.html", message=message, result=result), status


def _decode_uploaded_image(uploaded_file):
    image_data = uploaded_file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_data) > MAX_UPLOAD_BYTES:
        raise ValueError("That image is too large. Choose an image up to 50 MB.")

    image = None
    try:
        image = Image.open(BytesIO(image_data))
        image.load()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        if image is not None:
            image.close()
        raise ValueError("The selected file is not a readable image.") from exc

    width, height = image.size
    if width * height > MAX_IMAGE_PIXELS:
        image.close()
        raise ValueError("That image has too many pixels. Choose an image under 20 megapixels.")

    try:
        image_data = BytesIO()
        image.convert("RGB").save(image_data, format="PNG")
        return image_data.getvalue()
    except (OSError, ValueError) as exc:
        raise ValueError("The selected file is not a readable image.") from exc
    finally:
        image.close()


@app.errorhandler(RequestEntityTooLarge)
def _file_too_large(_error):
    return _render("The upload is too large. Choose images totaling up to 50 MB.", status=413)


@app.get("/")
def index():
    return _render()


@app.post("/predict")
def predict():
    side_files = {
        side: request.files.get(side)
        for side in ("front", "back", "left", "right")
    }
    has_side_upload = any(file and file.filename for file in side_files.values())
    if has_side_upload:
        missing_sides = [
            side for side, file in side_files.items() if file is None or not file.filename
        ]
        if missing_sides:
            return _render(
                f"Choose an image for every side. Missing: {', '.join(missing_sides)}.",
                status=400,
            )
        try:
            side_images = {
                side: _decode_uploaded_image(side_files[side])
                for side in use_model.CAMERA_SIDES
            }
        except ValueError as exc:
            return _render(str(exc), status=400)
    else:
        uploaded_file = request.files.get("image")
        if uploaded_file is None or not uploaded_file.filename:
            return _render("Choose an image for each side to continue.", status=400)
        try:
            image_data = _decode_uploaded_image(uploaded_file)
        except ValueError as exc:
            return _render(str(exc), status=400)

    try:
        with tempfile.TemporaryDirectory(prefix="mango-grade-") as temporary_directory:
            validity_model, grade_model, class_names, validity_names = _get_models()
            if has_side_upload:
                image_paths = {}
                for side in use_model.CAMERA_SIDES:
                    image_path = Path(temporary_directory) / f"{side}.png"
                    image_path.write_bytes(side_images[side])
                    image_paths[side] = image_path
                per_side = {
                    side: use_model.predict_single_image(
                        validity_model,
                        grade_model,
                        class_names,
                        image_paths[side],
                        validity_class_names=validity_names,
                    )
                    for side in use_model.CAMERA_SIDES
                }
                result = use_model.combine_view_predictions(per_side, class_names)
                if result["status"] != "NON_MANGO":
                    class_scores = {
                        name: sum(
                            per_side[side]["class_scores"][name]
                            for side in use_model.CAMERA_SIDES
                        )
                        / len(use_model.CAMERA_SIDES)
                        for name in class_names
                    }
                    result["class_scores"] = class_scores
                    result["grade_confidence"] = class_scores[result["predicted_grade"]]
                    result["per_side"] = per_side
            else:
                image_path = Path(temporary_directory) / "upload.png"
                image_path.write_bytes(image_data)
                result = use_model.predict_single_image(
                    validity_model,
                    grade_model,
                    class_names,
                    image_path,
                    validity_class_names=validity_names,
                )
    except FileNotFoundError as exc:
        app.logger.warning("Prediction model or class mapping is unavailable: %s", exc)
        return _render(
            "Grading is unavailable because the trained grade model or its class mapping is missing.",
            status=503,
        )
    except (ValueError, RuntimeError) as exc:
        app.logger.exception("Could not run image grading")
        return _render(f"Could not grade this image: {exc}", status=422)
    except Exception:
        app.logger.exception("Unexpected error while grading an uploaded image")
        return _render("Something went wrong while grading the image. Check the server log.", status=500)

    return _render(result=result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=5000, debug=False)
