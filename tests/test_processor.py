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


if __name__ == "__main__":
    unittest.main()
