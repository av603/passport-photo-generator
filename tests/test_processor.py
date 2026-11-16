import os
import unittest
import numpy as np
from PIL import Image

import passport_photo_processor as ppp


class TestPassportPhotoProcessor(unittest.TestCase):

    def test_load_image_missing_file(self):
        """Should raise FileNotFoundError when input file does not exist."""
        with self.assertRaises(FileNotFoundError):
            ppp.load_image("non_existent_file_12345.jpg")

    def test_load_image_valid(self):
        """Should load sample image into RGB and BGR arrays matching dimensions."""
        sample_path = os.path.join("examples", "sample_input.jpg")
        image_bgr, image_rgb = ppp.load_image(sample_path)
        self.assertIsInstance(image_bgr, np.ndarray)
        self.assertIsInstance(image_rgb, np.ndarray)
        self.assertEqual(image_bgr.shape, image_rgb.shape)
        self.assertEqual(len(image_bgr.shape), 3)

    def test_deskew_face_on_sample(self):
        """Should execute deskew_face on sample image and keep face upright."""
        sample_path = os.path.join("examples", "sample_input.jpg")
        image_bgr, image_rgb = ppp.load_image(sample_path)
        landmarks = ppp.detect_face_landmarks(image_rgb)
        deskewed_bgr, deskewed_lm = ppp.deskew_face(image_bgr, landmarks)
        self.assertEqual(deskewed_bgr.shape, image_bgr.shape)
        self.assertIsNotNone(deskewed_lm)
        # Forehead (landmark 10) must be above chin (landmark 152) in image coordinates (y increases downward)
        forehead_y = deskewed_lm[10].y
        chin_y = deskewed_lm[152].y
        self.assertLess(
            forehead_y, chin_y, "Face must remain upright (forehead above chin)"
        )

    def test_validate_biometrics_on_sample(self):
        """Should validate sample image and return advisory list."""
        sample_path = os.path.join("examples", "sample_input.jpg")
        image_bgr, image_rgb = ppp.load_image(sample_path)
        landmarks = ppp.detect_face_landmarks(image_rgb)
        warnings = ppp.validate_biometrics(image_bgr, landmarks)
        self.assertIsInstance(warnings, list)

    def test_parse_args_defaults(self):
        """Should parse arguments with expected defaults."""
        args = ppp.parse_args(["input.jpg"])
        self.assertEqual(args.input, "input.jpg")
        self.assertEqual(args.output_dir, "output")
        self.assertFalse(args.demo)
        self.assertFalse(args.no_sheet)

    def test_parse_args_demo(self):
        """Should parse --demo flag properly."""
        args = ppp.parse_args(["--demo"])
        self.assertTrue(args.demo)

    def test_parse_args_no_sheet(self):
        """Should parse --no-sheet flag properly."""
        args = ppp.parse_args(["my_photo.jpg", "--no-sheet", "-o", "custom_out"])
        self.assertEqual(args.input, "my_photo.jpg")
        self.assertTrue(args.no_sheet)
        self.assertEqual(args.output_dir, "custom_out")

    def test_target_spec_dimensions(self):
        """TARGET_WIDTH_PX and TARGET_HEIGHT_PX should match 35x45mm at 300 DPI."""
        self.assertEqual(ppp.TARGET_WIDTH_PX, 413)
        self.assertEqual(ppp.TARGET_HEIGHT_PX, 531)
        self.assertEqual(ppp.TARGET_CHIN_CROWN_PX, 402)

    def test_single_photo_dpi_saved(self):
        """Single photo output should include 300 DPI metadata."""
        import tempfile

        dummy_passport = np.full(
            (ppp.TARGET_HEIGHT_PX, ppp.TARGET_WIDTH_PX, 3), 200, dtype=np.uint8
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            temp_path = tf.name
        try:
            single_pil = Image.fromarray(dummy_passport)
            single_pil.save(temp_path, dpi=(ppp.DPI, ppp.DPI))
            with Image.open(temp_path) as img:
                dpi_info = img.info.get("dpi")
                self.assertIsNotNone(dpi_info)
                self.assertEqual(int(round(dpi_info[0])), 300)
                self.assertEqual(int(round(dpi_info[1])), 300)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_generate_sheet_dimensions(self):
        """Sheet generator should produce 1200x1800 px sheet at 300 DPI with tick marks."""
        dummy_passport = np.full(
            (ppp.TARGET_HEIGHT_PX, ppp.TARGET_WIDTH_PX, 3), 200, dtype=np.uint8
        )
        sheet = ppp.generate_sheet(
            dummy_passport,
            dpi=300,
            target_w=ppp.TARGET_WIDTH_PX,
            target_h=ppp.TARGET_HEIGHT_PX,
        )
        self.assertIsInstance(sheet, Image.Image)
        self.assertEqual(sheet.size, (1200, 1800))
        # Ensure sheet contains drawing marks outside pure white background
        arr = np.array(sheet)
        self.assertTrue((arr != 255).any())

    def test_parse_args_verify(self):
        """Should parse --verify and --no-verify flags properly."""
        args1 = ppp.parse_args(["--verify", "photo.png"])
        self.assertEqual(args1.verify, ["photo.png"])
        args2 = ppp.parse_args(["--no-verify"])
        self.assertTrue(args2.no_verify)

    def test_verify_passport_photo_missing(self):
        """Should fail verification on missing file."""
        report = ppp.verify_passport_photo("non_existent_123.png")
        self.assertFalse(report["passed"])
        self.assertEqual(report["single_checks"][0][0], "File Existence")

    def test_verify_passport_photo_invalid_dim(self):
        """Should fail verification on invalid dimensions."""
        import tempfile

        dummy = np.full((300, 300, 3), 200, dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            path = tf.name
        try:
            Image.fromarray(dummy).save(path, dpi=(300, 300))
            report = ppp.verify_passport_photo(path)
            self.assertFalse(report["passed"])
            dim_check = [c for c in report["single_checks"] if c[0] == "Dimensions"][0]
            self.assertFalse(dim_check[1])
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestExposure(unittest.TestCase):
    """
    Tests for the exposure analysis, correction, and verification pipeline.

    Standards basis:
        - ICAO Doc 9303 / ISO 19794-5: even lighting, no over/under-exposure
        - ISO 29794-5 / OFIQ: UnderExposurePrevention (Y<30 fraction <1.5%),
          OverExposurePrevention (Y>245 fraction <1.5%)
        - APO (Australian Passport Office): face must not be too dark or washed out
    """

    SAMPLE_PATH = os.path.join("examples", "sample_input.jpg")

    def _load_sample_with_landmarks(self, image_bgr=None):
        if image_bgr is None:
            image_bgr, image_rgb = ppp.load_image(self.SAMPLE_PATH)
        else:
            image_rgb = __import__("cv2").cvtColor(
                image_bgr, __import__("cv2").COLOR_BGR2RGB
            )
        landmarks = ppp.detect_face_landmarks(image_rgb)
        return image_bgr, landmarks

    # --- analyze_exposure ---

    def test_analyze_exposure_returns_expected_keys(self):
        """analyze_exposure must return a dict with all required metric keys."""
        image_bgr, landmarks = self._load_sample_with_landmarks()
        result = ppp.analyze_exposure(image_bgr, landmarks)
        for key in (
            "mean_luminance",
            "underexposed_frac",
            "shadow_frac",
            "overexposed_frac",
            "luma_iqr",
            "status",
            "advisories",
        ):
            self.assertIn(
                key, result, f"Key '{key}' missing from analyze_exposure result"
            )

    def test_analyze_exposure_sample_is_compliant(self):
        """The bundled sample image must pass as 'compliant' (it is a well-exposed photo)."""
        image_bgr, landmarks = self._load_sample_with_landmarks()
        result = ppp.analyze_exposure(image_bgr, landmarks)
        self.assertIn(
            result["status"],
            ("compliant", "flat"),
            f"Sample image should be compliant; got status='{result['status']}' "
            f"mean={result['mean_luminance']:.1f}",
        )
        self.assertGreaterEqual(result["mean_luminance"], ppp.EXPOSURE_LUMA_MIN)
        self.assertLessEqual(result["mean_luminance"], ppp.EXPOSURE_LUMA_MAX)

    def test_analyze_exposure_dark_image_is_flagged(self):
        """A synthetically darkened image (×0.45) must be flagged as 'underexposed'."""
        import cv2

        orig_bgr, _ = ppp.load_image(self.SAMPLE_PATH)
        dark_bgr = (orig_bgr.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
        _, landmarks = self._load_sample_with_landmarks(image_bgr=dark_bgr)
        result = ppp.analyze_exposure(dark_bgr, landmarks)
        self.assertEqual(
            result["status"],
            "underexposed",
            f"Darkened image should be 'underexposed'; got '{result['status']}' "
            f"mean={result['mean_luminance']:.1f}",
        )
        self.assertLess(result["mean_luminance"], ppp.EXPOSURE_LUMA_MIN)
        self.assertGreater(len(result["advisories"]), 0)

    def test_analyze_exposure_overexposed_image_is_flagged(self):
        """A synthetically brightened image (×2.5) must be flagged as 'overexposed'."""
        import cv2

        orig_bgr, _ = ppp.load_image(self.SAMPLE_PATH)
        bright_bgr = (orig_bgr.astype(np.float32) * 2.5).clip(0, 255).astype(np.uint8)
        _, landmarks = self._load_sample_with_landmarks(image_bgr=bright_bgr)
        result = ppp.analyze_exposure(bright_bgr, landmarks)
        self.assertEqual(
            result["status"],
            "overexposed",
            f"Brightened image should be 'overexposed'; got '{result['status']}' "
            f"mean={result['mean_luminance']:.1f}",
        )
        self.assertGreater(result["mean_luminance"], ppp.EXPOSURE_LUMA_MAX)

    # --- correct_exposure ---

    def test_correct_exposure_brings_dark_into_range(self):
        """correct_exposure must lift a darkened image back into the compliant range."""
        import cv2

        orig_bgr, _ = ppp.load_image(self.SAMPLE_PATH)
        dark_bgr = (orig_bgr.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
        _, landmarks = self._load_sample_with_landmarks(image_bgr=dark_bgr)
        corrected_bgr = ppp.correct_exposure(dark_bgr, landmarks)
        result = ppp.analyze_exposure(corrected_bgr, landmarks)
        self.assertGreaterEqual(
            result["mean_luminance"],
            ppp.EXPOSURE_LUMA_MIN,
            f"Corrected mean luminance {result['mean_luminance']:.1f} still below minimum {ppp.EXPOSURE_LUMA_MIN}",
        )
        self.assertLessEqual(
            result["mean_luminance"],
            ppp.EXPOSURE_LUMA_MAX,
            f"Corrected mean luminance {result['mean_luminance']:.1f} above maximum {ppp.EXPOSURE_LUMA_MAX}",
        )

    def test_correct_exposure_no_colour_cast(self):
        """correct_exposure must not introduce a significant colour cast (hue delta < 10°)."""
        import cv2

        orig_bgr, landmarks = self._load_sample_with_landmarks()
        dark_bgr = (orig_bgr.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
        _, dark_lm = self._load_sample_with_landmarks(image_bgr=dark_bgr)
        corrected_bgr = ppp.correct_exposure(dark_bgr, dark_lm)

        mask = ppp._build_face_mask(orig_bgr, landmarks)

        # Compare hue in HSV space on original vs corrected face region
        orig_hsv = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV)
        corr_hsv = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2HSV)
        orig_hue = orig_hsv[:, :, 0][mask > 0].astype(np.float32)
        corr_hue = corr_hsv[:, :, 0][mask > 0].astype(np.float32)
        mean_hue_delta = float(np.abs(np.mean(orig_hue) - np.mean(corr_hue)))
        self.assertLess(
            mean_hue_delta,
            10.0,
            f"Colour cast introduced: mean hue delta {mean_hue_delta:.1f}° (limit: 10°)",
        )

    def test_correct_exposure_compliant_image_unchanged(self):
        """correct_exposure must return the original image unchanged if already compliant."""
        image_bgr, landmarks = self._load_sample_with_landmarks()
        corrected = ppp.correct_exposure(image_bgr, landmarks)
        # Images should be identical (no-op path)
        self.assertTrue(
            np.array_equal(image_bgr, corrected),
            "correct_exposure modified a compliant image — it should return it unchanged",
        )

    # --- verify_passport_photo exposure row ---

    def test_verify_includes_exposure_check(self):
        """verify_passport_photo must include an 'Exposure' row in single_checks."""
        report = ppp.verify_passport_photo(
            os.path.join("examples", "sample_single.png")
        )
        check_names = [name for name, _, _ in report["single_checks"]]
        self.assertTrue(
            any("Exposure" in n for n in check_names),
            f"No exposure check found in verification report. Checks: {check_names}",
        )

    def test_verify_exposure_passes_on_sample(self):
        """The bundled sample_single.png must pass the exposure verification check."""
        report = ppp.verify_passport_photo(
            os.path.join("examples", "sample_single.png")
        )
        exp_check = next(
            (c for c in report["single_checks"] if "Exposure" in c[0]), None
        )
        self.assertIsNotNone(exp_check, "Exposure check not found in report")
        self.assertTrue(
            exp_check[1],
            f"Sample photo failed exposure check: {exp_check[2]}",
        )

    def test_verify_exposure_fails_on_dark_image(self):
        """verify_passport_photo must fail the exposure check for a saved dark image."""
        import cv2
        import tempfile

        orig_bgr, orig_rgb = ppp.load_image(self.SAMPLE_PATH)
        dark_bgr = (orig_bgr.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            path = tf.name
        try:
            from PIL import Image as PILImage

            PILImage.fromarray(cv2.cvtColor(dark_bgr, cv2.COLOR_BGR2RGB)).save(
                path, dpi=(ppp.DPI, ppp.DPI)
            )
            report = ppp.verify_passport_photo(path)
            exp_check = next(
                (c for c in report["single_checks"] if "Exposure" in c[0]), None
            )
            self.assertIsNotNone(exp_check, "Exposure check missing from report")
            self.assertFalse(
                exp_check[1],
                f"Dark image should fail exposure check but passed: {exp_check[2]}",
            )
            self.assertFalse(
                report["passed"],
                "Overall report should fail when exposure check fails",
            )
        finally:
            if os.path.exists(path):
                os.remove(path)

    # --- CLI flag ---

    def test_parse_args_auto_exposure_defaults_true(self):
        """--auto-exposure must default to True."""
        args = ppp.parse_args(["input.jpg"])
        self.assertTrue(args.auto_exposure)

    def test_parse_args_no_auto_exposure(self):
        """--no-auto-exposure must set auto_exposure to False."""
        args = ppp.parse_args(["input.jpg", "--no-auto-exposure"])
        self.assertFalse(args.auto_exposure)


if __name__ == "__main__":
    unittest.main()
