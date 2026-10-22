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


def _get_models() -> tuple[object | None, object, list[str], list[str] | None]:
    global _loaded_models
    if _loaded_models is None:
        with _model_lock:
            if _loaded_models is None:
                grade_model = use_model.load_grade_model()
                class_names = use_model.load_class_names()
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
    return _loaded_models


def _render(message: str | None = None, result: dict | None = None, status: int = 200):
    return render_template("index.html", message=message, result=result), status


@app.errorhandler(RequestEntityTooLarge)
def _file_too_large(_error):
    return _render("That image is too large. Choose an image up to 50 MB.", status=413)


@app.get("/")
def index():
    return _render()


@app.post("/predict")
def predict():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return _render("Choose an image to continue.", status=400)

    image_data = uploaded_file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_data) > MAX_UPLOAD_BYTES:
        return _render("That image is too large. Choose an image up to 50 MB.", status=413)

    try:
        with Image.open(BytesIO(image_data)) as image:
            width, height = image.size
            image.verify()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError):
        return _render("The selected file is not a readable image.", status=400)

    if width * height > MAX_IMAGE_PIXELS:
        return _render("That image has too many pixels. Choose an image under 20 megapixels.", status=400)

    try:
        with tempfile.TemporaryDirectory(prefix="mango-grade-") as temporary_directory:
            image_path = Path(temporary_directory) / "upload.png"
            with Image.open(BytesIO(image_data)) as image:
                image.convert("RGB").save(image_path, format="PNG")
            validity_model, grade_model, class_names, validity_names = _get_models()
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
