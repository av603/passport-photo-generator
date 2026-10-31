import sys
import os
import argparse
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Australian Passport Specifications at 300 DPI
DPI = 300
MM_TO_INCH = 1 / 25.4

# Official Australian standard dimensions (35mm width x 45mm height at 300 DPI)
TARGET_WIDTH_PX = int(round(35 * MM_TO_INCH * DPI))   # 413 pixels
TARGET_HEIGHT_PX = int(round(45 * MM_TO_INCH * DPI))  # 531 pixels

# Target chin-to-crown distance (34mm, center of official 32-36mm range)
TARGET_CHIN_CROWN_PX = int(round(34 * MM_TO_INCH * DPI)) # 402 pixels


def load_image(input_path):
    """Load an image from disk, auto-orient according to EXIF tags, and return BGR and RGB arrays."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find input image at {input_path}")
    try:
        with Image.open(input_path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            image_rgb = np.array(pil_img)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            return image_bgr, image_rgb
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise ValueError(f"Could not decode image from {input_path}: {e}")


def detect_face_landmarks(image_rgb):
    """Detect facial landmarks using MediaPipe Face Mesh."""
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            raise ValueError("No face detected in the image. Please provide a clear front-facing photo.")
        return results.multi_face_landmarks[0].landmark


def deskew_face(image_bgr, landmarks):
    """Rotate image so the eyes lie on a horizontal plane."""
    h, w, _ = image_bgr.shape
    left_eye = landmarks[468]   # Subject's left eye (viewer's right)
    right_eye = landmarks[473]  # Subject's right eye (viewer's left)

    lx, ly = left_eye.x * w, left_eye.y * h
    rx, ry = right_eye.x * w, right_eye.y * h

    dx = lx - rx
    dy = ly - ry
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))

    if abs(angle_deg) < 0.2:
        return image_bgr, landmarks

    eye_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)
    rot_matrix = cv2.getRotationMatrix2D(eye_center, angle_deg, 1.0)
    rotated_bgr = cv2.warpAffine(
        image_bgr, rot_matrix, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )
    rotated_rgb = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB)
    new_landmarks = detect_face_landmarks(rotated_rgb)
    return rotated_bgr, new_landmarks


def align_and_crop(image_bgr, face_landmarks, target_w=TARGET_WIDTH_PX, target_h=TARGET_HEIGHT_PX, target_cc=TARGET_CHIN_CROWN_PX):
    """Scale and crop face to meet passport dimensions."""
    h, w, _ = image_bgr.shape

    # Chin landmark: 152
    chin = face_landmarks[152]
    chin_px = np.array([chin.x * w, chin.y * h])

    # Eye center landmarks (left eye 468, right eye 473)
    left_eye = face_landmarks[468]
    right_eye = face_landmarks[473]
    eye_center_px = np.array([
        (left_eye.x + right_eye.x) * w / 2.0,
        (left_eye.y + right_eye.y) * h / 2.0
    ])

    # Anatomical estimation: Chin-to-Eye distance ≈ Eye-to-Crown distance
    chin_to_eye_dist = np.linalg.norm(chin_px - eye_center_px)
    vector_eye_to_chin = (chin_px - eye_center_px) / np.linalg.norm(chin_px - eye_center_px)
    crown_px = eye_center_px - vector_eye_to_chin * (chin_to_eye_dist * 1.05)

    current_cc_dist = np.linalg.norm(chin_px - crown_px)
    scale = target_cc / current_cc_dist

    # Resize image based on scale
    scaled_w = int(w * scale)
    scaled_h = int(h * scale)
    scaled_image = cv2.resize(image_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

    eye_center_scaled = eye_center_px * scale
    crown_scaled = crown_px * scale
    chin_scaled = chin_px * scale

    # True centering coordinates
    face_center_x = eye_center_scaled[0]
    face_center_y = (crown_scaled[1] + chin_scaled[1]) / 2.0

    # Crop calculation (centered both horizontally and vertically)
    crop_x1 = int(face_center_x - target_w / 2)
    crop_y1 = int(face_center_y - target_h / 2)

    crop_x2 = crop_x1 + target_w
    crop_y2 = crop_y1 + target_h

    # Pad scaled image if crop box goes out of bounds
    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - scaled_w)
    pad_bottom = max(0, crop_y2 - scaled_h)

    padded_image = cv2.copyMakeBorder(
        scaled_image, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    crop_x1 += pad_left
    crop_y1 += pad_top
    crop_x2 += pad_left
    crop_y2 += pad_top

    passport_cropped = padded_image[crop_y1:crop_y2, crop_x1:crop_x2]
    passport_cropped = cv2.resize(passport_cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return passport_cropped


def generate_sheet(passport_bgr, dpi=DPI, target_w=TARGET_WIDTH_PX, target_h=TARGET_HEIGHT_PX):
    """Tile passport photo into a 4x6 inch print sheet with scale bars."""
    sheet_w = int(4 * dpi) # 1200 px
    sheet_h = int(6 * dpi) # 1800 px
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    bar_len_px = int(50 * MM_TO_INCH * dpi)
    bar_thickness = 4

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    text_str = "50 mm"
    try:
        text_bbox = font.getbbox(text_str)
        text_w = text_bbox[2] - text_bbox[0]
        text_h_height = text_bbox[3] - text_bbox[1]
    except AttributeError:
        text_w, text_h_height = font.getsize(text_str)

    # 1. Horizontal Scale Bar
    hx1 = (sheet_w - bar_len_px) // 2
    hy1 = sheet_h - 45
    hx2 = hx1 + bar_len_px
    hy2 = hy1

    draw.line([(hx1, hy1 - 8), (hx1, hy1 + 8)], fill=(30, 30, 30), width=3)
    draw.line([(hx2, hy2 - 8), (hx2, hy2 + 8)], fill=(30, 30, 30), width=3)
    draw.line([(hx1, hy1), (hx2, hy2)], fill=(30, 30, 30), width=bar_thickness)

    mid_x = hx1 + (bar_len_px // 2)
    draw.line([(mid_x, hy1 - 5), (mid_x, hy1 + 5)], fill=(30, 30, 30), width=2)
    draw.text((mid_x - (text_w // 2), hy1 - text_h_height - 10), text_str, fill=(30, 30, 30), font=font)

    # 2. Vertical Scale Bar
    vx1 = sheet_w - 45
    vy1 = (sheet_h - bar_len_px) // 2
    vx2 = vx1
    vy2 = vy1 + bar_len_px

    draw.line([(vx1 - 8, vy1), (vx1 + 8, vy1)], fill=(30, 30, 30), width=3)
    draw.line([(vx2 - 8, vy2), (vx2 + 8, vy2)], fill=(30, 30, 30), width=3)
    draw.line([(vx1, vy1), (vx2, vy2)], fill=(30, 30, 30), width=bar_thickness)

    mid_y = vy1 + (bar_len_px // 2)
    draw.line([(vx1 - 5, mid_y), (vx1 + 5, mid_y)], fill=(30, 30, 30), width=2)
    draw.text((vx1 - (text_w // 2), vy1 - text_h_height - 10), text_str, fill=(30, 30, 30), font=font)

    # Paste photos
    passport_pil = Image.fromarray(cv2.cvtColor(passport_bgr, cv2.COLOR_BGR2RGB))
    cols = 2
    rows = 3
    margin_x = (sheet_w - (cols * target_w)) // (cols + 1)
    margin_y = (sheet_h - (rows * target_h)) // (rows + 1)

    for r in range(rows):
        for c in range(cols):
            x = margin_x + c * (target_w + margin_x)
            y = margin_y + r * (target_h + margin_y)
            sheet.paste(passport_pil, (x, y))

    return sheet


def process_passport_photo(input_path, output_single_path, output_sheet_path=None):
    """Main processing function to load, crop, and save passport photo and optional sheet."""
    image_bgr, image_rgb = load_image(input_path)
    landmarks = detect_face_landmarks(image_rgb)
    deskewed_bgr, deskewed_landmarks = deskew_face(image_bgr, landmarks)
    cropped_bgr = align_and_crop(deskewed_bgr, deskewed_landmarks)

    if output_single_path:
        out_dir = os.path.dirname(output_single_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        single_pil = Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))
        single_pil.save(output_single_path, dpi=(DPI, DPI))
        print(f"Saved single passport photo to {output_single_path}")

    if output_sheet_path:
        out_dir = os.path.dirname(output_sheet_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        sheet_pil = generate_sheet(cropped_bgr)
        sheet_pil.save(output_sheet_path, "PNG", dpi=(DPI, DPI))
        print(f"Saved 4x6 print sheet to {output_sheet_path}")

    return cropped_bgr


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Australian Passport Photo Generator — Crop, scale, align, and tile passport photos."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to input photo (e.g. input.jpg)."
    )
    parser.add_argument(
        "output_single",
        nargs="?",
        default=None,
        help="Optional path for single cropped photo output."
    )
    parser.add_argument(
        "output_sheet",
        nargs="?",
        default=None,
        help="Optional path for 4x6 print sheet output."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run immediately using bundled sample photo (examples/sample_input.jpg)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Output directory when specific file paths are omitted (default: output)."
    )
    parser.add_argument(
        "--no-sheet",
        action="store_true",
        help="Generate only the single passport photo, skipping the 4x6 sheet."
    )
    return parser.parse_args(args)


def main():
    args = parse_args()

    # Determine input path
    if args.demo:
        input_file = os.path.join("examples", "sample_input.jpg")
        if not os.path.exists(input_file):
            print(f"Error: Demo image not found at {input_file}")
            sys.exit(1)
        print("Running in DEMO mode with bundled sample image...")
    elif args.input:
        input_file = args.input
    else:
        # Check if legacy input.jpg exists in root
        if os.path.exists("input.jpg"):
            input_file = "input.jpg"
        else:
            print("Error: No input image specified. Provide an image path or use --demo.")
            print("Usage: python passport_photo_processor.py <input.jpg> [options]")
            print("       python passport_photo_processor.py --demo")
            sys.exit(1)

    # Determine output paths
    if args.output_single:
        output_single = args.output_single
    else:
        output_single = os.path.join(args.output_dir, "passport_single.png")

    if args.no_sheet:
        output_sheet = None
    elif args.output_sheet:
        output_sheet = args.output_sheet
    else:
        output_sheet = os.path.join(args.output_dir, "passport_sheet_4x6.png")

    try:
        process_passport_photo(input_file, output_single, output_sheet)
    except Exception as e:
        print(f"Processing Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
