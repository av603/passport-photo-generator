# Australian Passport Photo Generator

An automated tool built in Python to align, scale, crop, and tile photos according to official **Australian Passport Photo Specifications**. It utilizes OpenCV and MediaPipe Face Mesh for precise landmark detection, auto-deskewing, and face alignment. Outputs are provided as high-quality PNGs with embedded 300 DPI metadata ready for standard $4'' \times 6''$ photo prints.

---

## Features

- **EXIF Auto-Orientation:** Automatically normalizes smartphone photos using EXIF orientation tags before processing.
- **Auto-Deskew & Eye Leveling:** Detects head tilt (roll) via eye landmarks and applies an affine rotation matrix so eyes are horizontally level.
- **Official Australian Scaling:**
  - Photo size: **35mm width × 45mm height** ($413 \times 531$ px at 300 DPI).
  - Vertical chin-to-crown length: **34mm** ($402$ px, centered within the official **32mm–36mm** requirement).
  - Embedded 300 DPI metadata in single photo output for accurate print scaling.
- **$4'' \times 6''$ Photo Sheet Tiling:**
  - Generates a $2 \times 3$ grid layout (6 photos) on a $1200 \times 1800$ px canvas at 300 DPI.
  - **Corner Tick Marks:** Subtle `L` cutting guides extending outward from each photo boundary for scissor or rotary trimmer alignment.
  - **Calibration Scale Bars:** 50mm horizontal and vertical scale bars drawn in clear sheet margins to verify physical printer scaling with a ruler.
- **Biometric Compliance Advisories:** Pre-flight checks analyzing head yaw, open eyes (EAR), neutral expression (mouth closed), minimum resolution, and camera framing padding.
- **Non-Root Docker Environment:** Pre-packaged dependencies running with non-root user permissions (`appuser` UID 1000) so generated files match host user ownership.

---

## Requirements & Specifications

### Australian Passport Photo Guidelines
- **Width:** 35mm to 40mm (standardized to **35mm** / $413$ px @ 300 DPI)
- **Height:** 45mm to 50mm (standardized to **45mm** / $531$ px @ 300 DPI)
- **Chin-to-Crown Distance:** 32mm to 36mm (target set to **34mm** / $402$ px)
- **Background:** Uniform light gray or white background recommended.

---

## Project Structure

```text
.
├── Dockerfile                  # Container setup with non-root user and fonts
├── requirements.txt            # Pinned dependencies
├── passport_photo_processor.py # Main processing engine and CLI
├── examples/                   # Bundled sample assets
│   ├── sample_input.jpg        # Demo portrait
│   ├── sample_single.png       # Reference 35x45mm single photo
│   └── sample_sheet_4x6.png    # Reference 4x6 print sheet
├── tests/                      # Automated unit tests
│   └── test_processor.py
├── output/                     # Default output destination (.gitignored)
└── README.md
```

---

## Quickstart with Docker

### 1. Build the Docker Image

```bash
docker build -t passport-photo-processor .
```

### 2. Verify with Out-of-the-Box Demo Mode

Run against the bundled sample photo with zero configuration:

```bash
docker run --rm -v "$(pwd)":/app passport-photo-processor --demo
```

Outputs will be saved to `output/passport_single.png` and `output/passport_sheet_4x6.png`.

### 3. Process Your Own Photo

#### Option A: Photo is in your project directory
Place your photo (e.g. `my_photo.jpg`) in the project directory and run:

```bash
docker run --rm -v "$(pwd)":/app passport-photo-processor my_photo.jpg
```

#### Option B: Photo is located anywhere on your computer (e.g. Downloads, Pictures)
Because Docker containers run with isolated filesystems, mount `$HOME` so any host path works transparently:

```bash
docker run --rm \
  -v "$HOME:$HOME" \
  -w "$(pwd)" \
  passport-photo-processor /home/av603/Downloads/my_photo.jpg
```

#### Option C: Convenient Shell Alias
Add this alias to your `~/.bashrc` or `~/.zshrc` to run the tool directly from anywhere on any photo:

```bash
alias passport-photo='docker run --rm -v "$HOME:$HOME" -w "$(pwd)" passport-photo-processor'
```

Then simply run:

```bash
passport-photo ~/Downloads/my_photo.jpg
```

---

## Command Line Interface (CLI)

```text
usage: passport_photo_processor.py [-h] [--demo] [-o OUTPUT_DIR] [--no-sheet]
                                   [input] [output_single] [output_sheet]

positional arguments:
  input                 Path to input photo (e.g. my_photo.jpg).
  output_single         Optional path for single cropped photo output.
  output_sheet          Optional path for 4x6 print sheet output.

options:
  -h, --help            Show this help message and exit.
  --demo                Run immediately using bundled sample photo.
  -o, --output-dir      Target directory for outputs (default: output).
  --no-sheet            Generate only the single passport photo, skipping the sheet.
```

---

## Local Development (Without Docker)

### Prerequisites

System dependencies (OpenCV runtime and fonts on Ubuntu / Linux Mint):

```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0 fonts-dejavu-core
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Running Tests

```bash
python3 -m unittest discover tests
```

### Running Locally

```bash
python3 passport_photo_processor.py --demo
python3 passport_photo_processor.py path/to/photo.jpg
```

---

## Outputs

1. **`output/passport_single.png`**: $413 \times 531$ px ($35\text{mm} \times 45\text{mm}$ @ 300 DPI) single crop with embedded DPI metadata.
2. **`output/passport_sheet_4x6.png`**: $1200 \times 1800$ px ($4'' \times 6''$ @ 300 DPI) ready to print at photo kiosks (e.g. Kmart, Harvey Norman, Officeworks) or home photo printers. Includes:
   - 6 correctly proportioned photos.
   - Subtle corner tick marks (`L`) to guide clean cuts with a ruler and scissors.
   - 50mm horizontal and vertical calibration bars. Measure with a physical ruler after printing: if the line measures exactly 50mm, your photos are 100% compliant in size.

---

## License & Attribution

- Source code licensed under the [MIT License](LICENSE).
- Sample demo portrait in `examples/sample_input.jpg` is provided for testing and demonstration purposes.
