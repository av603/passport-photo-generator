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
        """Should execute deskew_face on sample image without error."""
        sample_path = os.path.join("examples", "sample_input.jpg")
        image_bgr, image_rgb = ppp.load_image(sample_path)
        landmarks = ppp.detect_face_landmarks(image_rgb)
        deskewed_bgr, deskewed_lm = ppp.deskew_face(image_bgr, landmarks)
        self.assertEqual(deskewed_bgr.shape, image_bgr.shape)
        self.assertIsNotNone(deskewed_lm)

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

    def test_generate_sheet_dimensions(self):
        """Sheet generator should produce 1200x1800 px sheet at 300 DPI."""
        dummy_passport = np.full((561, 442, 3), 200, dtype=np.uint8)
        sheet = ppp.generate_sheet(dummy_passport, dpi=300, target_w=442, target_h=561)
        self.assertIsInstance(sheet, Image.Image)
        self.assertEqual(sheet.size, (1200, 1800))


if __name__ == "__main__":
    unittest.main()
