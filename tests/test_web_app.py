from __future__ import annotations

from io import BytesIO
import unittest
from unittest.mock import patch

from PIL import Image

import web_app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        web_app.app.config["TESTING"] = True
        self.client = web_app.app.test_client()

    def test_home_page_displays_image_upload_form(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'name="image"', response.data)
        self.assertIn(b"Detect grade", response.data)
        self.assertIn(b"WebP, BMP, or TIFF", response.data)
        self.assertIn(b"up to 50 MB", response.data)

    def test_image_upload_returns_grade_and_model_scores(self):
        image_bytes = BytesIO()
        Image.new("RGB", (20, 12), color=(220, 120, 30)).save(image_bytes, format="JPEG")
        result = {
            "status": "VALIDITY_UNAVAILABLE",
            "predicted_grade": "grade 3",
            "grade_confidence": 0.82,
            "class_scores": {"grade 1": 0.01, "grade 3": 0.82, "grade 4": 0.17},
        }
        with (
            patch.object(web_app, "_get_models", return_value=(None, object(), ["grade 1", "grade 3", "grade 4"], None)),
            patch.object(web_app.use_model, "predict_single_image", return_value=result) as predict_image,
        ):
            response = self.client.post(
                "/predict",
                data={"image": (BytesIO(image_bytes.getvalue()), "mango.jpg")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Grade 3", response.data)
        self.assertIn(b"82.0%", response.data)
        self.assertIn(b"Mango validation is not configured", response.data)
        self.assertEqual(predict_image.call_args.args[3].name, "upload.png")
        self.assertFalse(predict_image.call_args.args[3].exists())

    def test_webp_upload_is_converted_to_rgb_png_for_prediction(self):
        image_bytes = BytesIO()
        Image.new("RGB", (20, 12), color=(220, 120, 30)).save(image_bytes, format="WEBP")
        result = {
            "status": "VALIDITY_UNAVAILABLE",
            "predicted_grade": "grade 2",
            "grade_confidence": 0.75,
            "class_scores": {"grade 2": 0.75},
        }
        with (
            patch.object(web_app, "_get_models", return_value=(None, object(), ["grade 2"], None)),
            patch.object(web_app.use_model, "predict_single_image", return_value=result) as predict_image,
        ):
            response = self.client.post(
                "/predict",
                data={"image": (BytesIO(image_bytes.getvalue()), "mango.webp")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Grade 2", response.data)
        predicted_path = predict_image.call_args.args[3]
        self.assertEqual(predicted_path.name, "upload.png")
        self.assertFalse(predicted_path.exists())

    def test_valid_mango_result_is_shown_explicitly(self):
        image_bytes = BytesIO()
        Image.new("RGB", (20, 12), color=(220, 120, 30)).save(image_bytes, format="PNG")
        result = {
            "status": "GRADED",
            "validity_status": "AVAILABLE",
            "validity_confidence": 0.96,
            "predicted_grade": "grade 2",
            "grade_confidence": 0.75,
            "class_scores": {"grade 2": 0.75},
        }
        with (
            patch.object(web_app, "_get_models", return_value=(object(), object(), ["grade 2"], ["Mango", "Non-Mango"])),
            patch.object(web_app.use_model, "predict_single_image", return_value=result),
        ):
            response = self.client.post(
                "/predict",
                data={"image": (BytesIO(image_bytes.getvalue()), "mango.png")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Mango check: Valid", response.data)
        self.assertIn(b"Grade 2", response.data)

    def test_other_readable_image_formats_are_accepted(self):
        result = {
            "status": "VALIDITY_UNAVAILABLE",
            "predicted_grade": "grade 2",
            "grade_confidence": 0.75,
            "class_scores": {"grade 2": 0.75},
        }
        for image_format in ("PNG", "BMP", "TIFF", "GIF"):
            with self.subTest(image_format=image_format):
                image_bytes = BytesIO()
                Image.new("RGB", (20, 12), color=(220, 120, 30)).save(
                    image_bytes, format=image_format
                )
                with (
                    patch.object(
                        web_app,
                        "_get_models",
                        return_value=(None, object(), ["grade 2"], None),
                    ),
                    patch.object(
                        web_app.use_model,
                        "predict_single_image",
                        return_value=result,
                    ),
                ):
                    response = self.client.post(
                        "/predict",
                        data={
                            "image": (
                                BytesIO(image_bytes.getvalue()),
                                f"mango.{image_format.lower()}",
                            )
                        },
                        content_type="multipart/form-data",
                    )

                self.assertEqual(response.status_code, 200)
                self.assertIn(b"Grade 2", response.data)

    def test_upload_rejects_non_image_content(self):
        response = self.client.post(
            "/predict",
            data={"image": (BytesIO(b"not an image"), "mango.jpg")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"not a readable image", response.data)

    def test_missing_grade_model_is_reported_to_user(self):
        image_bytes = BytesIO()
        Image.new("RGB", (20, 12), color=(220, 120, 30)).save(image_bytes, format="PNG")
        with patch.object(web_app, "_get_models", side_effect=FileNotFoundError("model missing")):
            response = self.client.post(
                "/predict",
                data={"image": (BytesIO(image_bytes.getvalue()), "mango.png")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn(b"trained grade model", response.data)

    def test_non_mango_result_does_not_show_grade_scores(self):
        image_bytes = BytesIO()
        Image.new("RGB", (20, 12), color=(220, 120, 30)).save(image_bytes, format="PNG")
        result = {
            "status": "NON_MANGO",
            "predicted_grade": None,
            "grade_confidence": None,
            "class_scores": {},
        }
        with (
            patch.object(web_app, "_get_models", return_value=(object(), object(), [], ["Mango", "Non-Mango"])),
            patch.object(web_app.use_model, "predict_single_image", return_value=result),
        ):
            response = self.client.post(
                "/predict",
                data={"image": (BytesIO(image_bytes.getvalue()), "object.png")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid for mango grading", response.data)
        self.assertIn(b"not recognized as a mango", response.data)
        self.assertNotIn(b"Top model score", response.data)
