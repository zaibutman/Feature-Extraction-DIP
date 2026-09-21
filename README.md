<div align="center">

# 🎯 Feature Extraction using Different Feature Descriptors

### A hands-on computer vision toolkit demonstrating 7 classic feature extraction techniques

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-Enabled-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-image.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=plotly&logoColor=white)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<p>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Course-Programming%20for%20AI-blueviolet?style=flat-square" />
  <img src="https://img.shields.io/badge/Made%20with-%E2%9D%A4-red?style=flat-square" />
</p>

</div>

---

## 📖 Overview

This project is a compact, self-contained demonstration of **seven classic feature descriptors** widely used in computer vision — from edge and corner detection to local keypoints, texture, shape, and color analysis.

It runs on a synthetically generated test image out of the box (so there's zero setup friction), but works just as well on **any image you provide** — `.jpg`, `.png`, `.bmp`, `.webp`, and more.

Every result is saved to an `output/` folder **and** automatically combined into a single preview grid that pops open in your file explorer / image viewer as soon as the script finishes — no digging through folders required.

---

## ✨ Features Covered

| # | Category | Technique | What it Detects |
|---|----------|-----------|------------------|
| 1️⃣ | **Edge Features** | Canny | Sharp intensity transitions / object boundaries |
| 2️⃣ | **Corner Features** | Harris | High-curvature points where edges meet |
| 3️⃣ | **Local / Blob Features** | SIFT | Scale-invariant keypoints & descriptors |
| 4️⃣ | **Local Binary Features** | ORB | Fast, rotation-invariant binary keypoints |
| 5️⃣ | **Texture Features** | LBP (Local Binary Pattern) | Local texture / micro-pattern structure |
| 6️⃣ | **Shape Features** | HOG (Histogram of Oriented Gradients) | Gradient-based object shape |
| 7️⃣ | **Color Features** | Color Histogram | Distribution of pixel intensities per channel |

---

## 🖼️ Sample Output

Running the script produces an all-in-one preview grid like this:

```
output/
├── 0_original_image.png
├── 1_canny_edges.png
├── 2_harris_corners.png
├── 3_sift_keypoints.png
├── 4_orb_keypoints.png
├── 5_lbp_texture.png
├── 6_hog_shape.png
├── 7_color_histogram.png
└── preview_all_results.png   ← combined grid, opened automatically
```

> 💡 *Tip: push your own `output/preview_all_results.png` to the repo and reference it here — e.g. `![Preview](output/preview_all_results.png)` — so it renders directly on this page.*

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-python scikit-image matplotlib numpy
```

### Run with the built-in test image

No image? No problem — the script auto-generates one with shapes and a checkerboard texture pattern:

```bash
python feature_descriptors.py
```

### Run with your own image

```bash
python feature_descriptors.py path/to/your_image.jpg
```

Supported formats include `.jpg`, `.png`, `.bmp`, `.tiff`, and `.webp`.

---

## ⚙️ How It Works

1. **Loads or generates** an image (`load_or_create_image`)
2. **Runs each descriptor** independently, printing key stats (edge pixel counts, keypoint counts, descriptor shapes, histogram bins, etc.) to the console
3. **Saves each result** as a standalone image inside `output/`
4. **Builds a combined preview grid** (`create_preview`) so all 8 results can be compared at a glance
5. **Auto-opens the results** — the preview image and the output folder — using the OS's native file handler (`os.startfile` / `open` / `xdg-open`, with a `webbrowser` fallback)

The `output/` directory is always created **next to the script itself**, regardless of the working directory the script is launched from — so results never get lost in an unrelated folder.

---

## 🗂️ Project Structure

```
.
├── feature_descriptors.py   # Main script — all 7 descriptors + preview logic
├── output/                  # Auto-generated results (created on first run)
└── README.md
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **OpenCV** — Canny, Harris, SIFT, ORB, color histograms
- **scikit-image** — Local Binary Pattern, Histogram of Oriented Gradients
- **Matplotlib** — plotting and the combined preview grid
- **NumPy** — array/image manipulation

---

## 👤 Author

**Shah Mubarak Zaib**
AI/ML Engineer · Final Year CS, Islamia College Peshawar

📘 *Assignment for the "Programming for AI" course.*

---

<div align="center">

⭐ If you found this useful, consider giving the repo a star!

</div>
