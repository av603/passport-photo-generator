# Australian Passport Photo Generator

An automated tool built in Python to align, scale, crop, and tile photos according to official **Australian Passport Photo Specifications**. It utilizes OpenCV and MediaPipe Face Mesh for precise landmark detection and face alignment, and output images are provided as high-quality PNGs ready for standard $4'' \times 6''$ photo prints.

---

## Features

- **Facial Landmark Detection:** Detects face orientation, chin position, and top of crown using MediaPipe Face Mesh.
- **Auto-Scale & Crop:** Adjusts dimensions to match Australian requirements:
  - Photo size: **35mm width × 45mm height** (at 300 DPI).
  - Vertical head length: **34mm** (target within the official **32mm–36mm** chin-to-crown requirement).
- **$4'' \times 6''$ Photo Sheet Tiling:** Generates a $2 \times 3$ grid layout (6 photos) on a $4'' \times 6''$ canvas, correctly spaced for print.
- **Dockerized Environment:** Pre-packaged dependencies ensure consistency on Linux Mint or any Docker-compatible environment.

---

## Requirements & Specifications

### Australian Passport Photo Guidelines
- **Width:** 35mm to 40mm (defaulted to 35mm)
- **Height:** 45mm to 50mm (defaulted to 45mm)
- **Chin-to-Crown Distance:** 32mm to 36mm (target set to 34mm)
- **Background:** Uniform light gray or white background recommended.

---

## Project Structure

```text
.
├── Dockerfile              # Docker container setup
├── passport_photo_processor.py   # Main Python processing script
├── README.md               # Documentation
├── input.jpg               # Your input photo (JPEG)
├── passport_single.png     # Generated single cropped photo
└── passport_sheet_4x6.png  # Generated 4x6 print sheet
```

---

## Quickstart with Docker

### 1. Build the Docker Image

Run the following command in your terminal from the project directory:

```bash
docker build -t passport-photo-processor .
```

### 2. Run the Container

Place your input photo (`input.jpg`) in the current directory and run:

```bash
docker run --rm -v "$(pwd)":/app passport-photo-processor input.jpg passport_single.png passport_sheet_4x6.png
```

#### Command Line Arguments
The script accepts optional arguments in order:
1. **Input file:** Path to input image (default: `input.jpg`)
2. **Single output:** Path for cropped single photo (default: `passport_single.png`)
3. **Sheet output:** Path for tiled 4x6 photo sheet (default: `passport_sheet_4x6.png`)

---

## Local Development (Optional)

If you wish to run the script locally on Linux Mint without Docker:

### Prerequisites

Install system dependencies and Python packages:

```bash
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libopencv-dev
pip install opencv-python-headless mediapipe Pillow numpy
```

### Execution

```bash
python3 passport_photo_processor.py input.jpg output_single.png output_4x6.png
```

---

## Outputs

1. **`passport_single.png`**: $413 \times 531$ px ($35\text{mm} \times 45\text{mm}$ @ 300 DPI) single crop.
2. **`passport_sheet_4x6.png`**: $1200 \times 1800$ px ($4'' \times 6''$ @ 300 DPI) ready to print at photo print shops or home photo printers.