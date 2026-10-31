import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# Australian Passport Specifications at 300 DPI
DPI = 300
MM_TO_INCH = 1 / 25.4

# Target dimensions set to the center of the official ranges (Width: 35-40mm, Height: 45-50mm)
TARGET_WIDTH_PX = int(37.5 * MM_TO_INCH * DPI)   # ~443 pixels
TARGET_HEIGHT_PX = int(47.5 * MM_TO_INCH * DPI)  # ~561 pixels

# Target chin-to-crown distance (aiming for 34mm, the exact center of the 32-36mm range)
TARGET_CHIN_CROWN_PX = int(34 * MM_TO_INCH * DPI) # ~402 pixels

def process_passport_photo(input_path, output_single_path, output_sheet_path):
    # Load image with OpenCV
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load input image from {input_path}")
    
    h, w, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe Face Mesh
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
        
        face_landmarks = results.multi_face_landmarks[0].landmark

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
        scale = TARGET_CHIN_CROWN_PX / current_cc_dist

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
        crop_x1 = int(face_center_x - TARGET_WIDTH_PX / 2)
        crop_y1 = int(face_center_y - TARGET_HEIGHT_PX / 2)

        crop_x2 = crop_x1 + TARGET_WIDTH_PX
        crop_y2 = crop_y1 + TARGET_HEIGHT_PX

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
        passport_cropped = cv2.resize(passport_cropped, (TARGET_WIDTH_PX, TARGET_HEIGHT_PX), interpolation=cv2.INTER_AREA)

        # Save single passport photo as PNG
        cv2.imwrite(output_single_path, passport_cropped)
        print(f"Saved single passport photo to {output_single_path}")

        # --- TILING ON 4"x6" PHOTO ---
        sheet_w = int(4 * DPI) # 1200 px
        sheet_h = int(6 * DPI) # 1800 px
        sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)

        # --- DRAW SYMMETRICALLY CENTERED SCALE BARS & TEXT BEHIND THE IMAGES ---
        bar_len_px = int(50 * MM_TO_INCH * DPI) # ~591 pixels for 50mm
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

        # 1. Horizontal Scale Bar (Bottom Center, perfectly centered on sheet width)
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

        # 2. Vertical Scale Bar (Right Center, perfectly centered on sheet height)
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

        # --- PASTE PASSPORT PHOTOS ON TOP ---
        passport_pil = Image.fromarray(cv2.cvtColor(passport_cropped, cv2.COLOR_BGR2RGB))

        cols = 2
        rows = 3
        margin_x = (sheet_w - (cols * TARGET_WIDTH_PX)) // (cols + 1)
        margin_y = (sheet_h - (rows * TARGET_HEIGHT_PX)) // (rows + 1)

        for r in range(rows):
            for c in range(cols):
                x = margin_x + c * (TARGET_WIDTH_PX + margin_x)
                y = margin_y + r * (TARGET_HEIGHT_PX + margin_y)
                sheet.paste(passport_pil, (x, y))

        sheet.save(output_sheet_path, "PNG", dpi=(DPI, DPI))
        print(f"Saved sheet with perfectly centered face cropping to {output_sheet_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python passport_photo_processor.py <input_jpg> [output_single.png] [output_sheet.png]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_single = sys.argv[2] if len(sys.argv) > 2 else "passport_single.png"
    output_sheet = sys.argv[3] if len(sys.argv) > 3 else "passport_sheet_4x6.png"

    process_passport_photo(input_file, output_single, output_sheet)