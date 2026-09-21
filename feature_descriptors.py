"""

Assignment:
Feature Extraction using Different Types of Feature Descriptors

Author: Shah Mubarak Zaib
Course: Programming for AI

About this program
------------------
This program demonstrates a few common feature descriptors used in
computer vision. I used a small generated image instead of depending on
an external image, which makes the program easier to run and test.

The program covers:

1. Edge features      -> Canny
2. Corner features    -> Harris
3. Local/Blob features-> SIFT
4. Local binary       -> ORB
5. Texture features   -> LBP
6. Shape features     -> HOG
7. Color features     -> Color Histogram

For each method, the program extracts the features and saves an image
showing the result in the "output" folder.

How to run
----------
Normal run:
    python feature_descriptors.py

Using your own image:
    python feature_descriptors.py path/to/your_image.jpg

Required libraries:
    pip install opencv-python scikit-image matplotlib numpy
"""

import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # lets the program save plots even without a GUI
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog


# Always save next to this script, regardless of what folder the program
# was launched from (VS Code, double-click, another terminal, etc.).
# This avoids output ending up somewhere unexpected, like an editor's
# install directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


# ---------------------------------------------------------------------------
# Step 0: Prepare the image
# ---------------------------------------------------------------------------
def load_or_create_image(path: str | None) -> np.ndarray:
    """
    If an image path is given, use that image.
    Otherwise, create a simple test image with a few different shapes
    and a checkerboard area for the texture example.
    """
    if path and os.path.exists(path):
        img = cv2.imread(path)
        print(f"[INFO] Loaded image: {path}")
        return img

    print("[INFO] No image was provided. Creating a test image...")

    # Start with a light background.
    img = np.full((400, 500, 3), 235, dtype=np.uint8)

    # Rectangle: useful for showing straight edges and corners.
    cv2.rectangle(img, (40, 40), (200, 180), (60, 60, 200), -1)

    # Circle: gives the image a rounded shape for local feature detection.
    cv2.circle(img, (350, 100), 70, (40, 160, 40), -1)

    # Triangle: adds more corners and a different orientation.
    pts = np.array([[300, 300], [400, 300], [350, 220]], np.int32)
    cv2.fillPoly(img, [pts], (200, 140, 20))

    # Checkerboard: included mainly to make the texture result easier to see.
    tex_x, tex_y, tex_size, cell = 40, 250, 140, 14
    for i in range(tex_size // cell):
        for j in range(tex_size // cell):
            color = (30, 30, 30) if (i + j) % 2 == 0 else (220, 220, 220)
            cv2.rectangle(
                img,
                (tex_x + i * cell, tex_y + j * cell),
                (tex_x + (i + 1) * cell, tex_y + (j + 1) * cell),
                color,
                -1
            )

    return img


def save(name: str, image) -> None:
    """Save one of the generated results in the output folder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(out_path, image)
    print(f"          -> saved: {os.path.abspath(out_path)}")


# ---------------------------------------------------------------------------
# 1. Edge features - Canny
# ---------------------------------------------------------------------------
def run_canny(gray):
    print("\n[1] EDGE FEATURES -> Canny")
    edges = cv2.Canny(gray, 100, 200)

    # Count the pixels that were detected as edges.
    edge_pixel_count = int(np.count_nonzero(edges))
    print(f"          Edge pixels detected : {edge_pixel_count}")

    save("1_canny_edges.png", edges)


# ---------------------------------------------------------------------------
# 2. Corner features - Harris
# ---------------------------------------------------------------------------
def run_harris(gray, color_img):
    print("[2] CORNER FEATURES -> Harris")

    gray_f = np.float32(gray)
    harris = cv2.cornerHarris(
        gray_f,
        blockSize=2,
        ksize=3,
        k=0.04
    )
    harris = cv2.dilate(harris, None)

    # Mark the stronger corner responses on the original image.
    result = color_img.copy()
    threshold = 0.01 * harris.max()
    result[harris > threshold] = [0, 0, 255]

    num_corners = int(np.sum(harris > threshold))
    print(f"          Strong corner points : {num_corners}")

    save("2_harris_corners.png", result)


# ---------------------------------------------------------------------------
# 3. Local / Blob features - SIFT
# ---------------------------------------------------------------------------
def run_sift(gray, color_img):
    print("[3] LOCAL / BLOB FEATURES -> SIFT")

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    print(f"          Keypoints found      : {len(keypoints)}")

    if descriptors is not None:
        print(
            f"          Descriptor shape     : {descriptors.shape} "
            "(128 values for each keypoint)"
        )

    # Draw the detected keypoints so we can see where SIFT found them.
    result = cv2.drawKeypoints(
        color_img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    save("3_sift_keypoints.png", result)


# ---------------------------------------------------------------------------
# 4. Local binary features - ORB
# ---------------------------------------------------------------------------
def run_orb(gray, color_img):
    print("[4] LOCAL BINARY FEATURES -> ORB")

    orb = cv2.ORB_create(nfeatures=300)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    print(f"          Keypoints found      : {len(keypoints)}")

    if descriptors is not None:
        print(
            f"          Descriptor shape     : {descriptors.shape} "
            "(32-byte binary descriptor)"
        )

    result = cv2.drawKeypoints(
        color_img,
        keypoints,
        None,
        color=(0, 255, 0)
    )

    save("4_orb_keypoints.png", result)


# ---------------------------------------------------------------------------
# 5. Texture features - LBP
# ---------------------------------------------------------------------------
def run_lbp(gray):
    print("[5] TEXTURE FEATURES -> Local Binary Pattern (LBP)")

    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        n_points,
        radius,
        method="uniform"
    )

    # Convert the LBP values to 8-bit so the result can be saved as an image.
    lbp_img = (lbp / lbp.max() * 255).astype(np.uint8)

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    print(f"          LBP histogram ({len(hist)} bins): {hist}")

    save("5_lbp_texture.png", lbp_img)


# ---------------------------------------------------------------------------
# 6. Shape features - HOG
# ---------------------------------------------------------------------------
def run_hog(gray):
    print("[6] SHAPE FEATURES -> Histogram of Oriented Gradients (HOG)")

    features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm="L2-Hys"
    )

    print(f"          HOG feature length   : {features.shape[0]}")

    # Scale the HOG visualization to normal image values before saving.
    hog_img_uint8 = (
        hog_image / (hog_image.max() + 1e-8) * 255
    ).astype(np.uint8)

    save("6_hog_shape.png", hog_img_uint8)


# ---------------------------------------------------------------------------
# 7. Bonus - Color Histogram
# ---------------------------------------------------------------------------
def run_color_histogram(color_img):
    print("[7] BONUS -> Color Histogram")

    colors = ("b", "g", "r")

    plt.figure(figsize=(6, 4))

    for i, col in enumerate(colors):
        hist = cv2.calcHist(
            [color_img],
            [i],
            None,
            [256],
            [0, 256]
        )
        plt.plot(hist, color=col)

    plt.title("Color Histogram (BGR channels)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "7_color_histogram.png")

    plt.savefig(out_path)
    plt.close()

    print(f"          -> saved: {os.path.abspath(out_path)}")


# ---------------------------------------------------------------------------
# Preview: combine every saved result into one image and open it
# ---------------------------------------------------------------------------
PREVIEW_FILES = [
    ("0_original_image.png", "Original"),
    ("1_canny_edges.png", "Canny (Edges)"),
    ("2_harris_corners.png", "Harris (Corners)"),
    ("3_sift_keypoints.png", "SIFT (Keypoints)"),
    ("4_orb_keypoints.png", "ORB (Keypoints)"),
    ("5_lbp_texture.png", "LBP (Texture)"),
    ("6_hog_shape.png", "HOG (Shape)"),
    ("7_color_histogram.png", "Color Histogram"),
]


def create_preview() -> str | None:
    """
    Build one combined image (a grid) out of all the saved results so the
    user can see everything at a glance instead of opening 8 separate files.
    Returns the path to the preview image, or None if it couldn't be built.
    """
    print("\n[PREVIEW] Building a combined preview image...")

    cols, rows = 4, 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    fig.suptitle("Feature Descriptor Results - Preview", fontsize=14)

    for ax, (filename, title) in zip(axes.ravel(), PREVIEW_FILES):
        path = os.path.join(OUTPUT_DIR, filename)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                # cv2 loads as BGR; matplotlib expects RGB.
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                continue
        ax.text(0.5, 0.5, "not found", ha="center", va="center")

    plt.tight_layout()
    preview_path = os.path.join(OUTPUT_DIR, "preview_all_results.png")
    plt.savefig(preview_path, dpi=120)
    plt.close(fig)

    preview_abspath = os.path.abspath(preview_path)
    print(f"          -> saved: {preview_abspath}")
    return preview_abspath


def open_path(path: str) -> bool:
    """
    Try to open a file or folder with whatever the operating system's
    default viewer/explorer is. Returns True if a viewer was launched
    without an immediate error, False otherwise (caller should then show
    the path so the user can open it manually).
    """
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
            return True
        elif sys.platform == "darwin":
            return os.system(f'open "{path}"') == 0
        else:
            return os.system(f'xdg-open "{path}" > /dev/null 2>&1') == 0
    except Exception:
        pass

    # Fallback: works in more edge cases (some WSL setups, remote
    # terminals, IDEs without a registered file-open handler).
    try:
        import webbrowser
        return webbrowser.open(f"file://{os.path.abspath(path)}")
    except Exception:
        return False


def open_containing_folder(path: str) -> bool:
    """
    Open the folder that contains `path` in the OS file explorer (not the
    file itself). This is a much more reliable fallback on Windows than
    trying to launch a specific file, since Explorer almost always exists
    and is registered, whereas a given file type might not have a
    default app, or the app might open minimized/behind other windows.
    """
    folder = os.path.dirname(os.path.abspath(path))
    try:
        if sys.platform.startswith("win"):
            os.startfile(folder)  # type: ignore[attr-defined]
            return True
        elif sys.platform == "darwin":
            return os.system(f'open "{folder}"') == 0
        else:
            return os.system(f'xdg-open "{folder}" > /dev/null 2>&1') == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    user_path = sys.argv[1] if len(sys.argv) > 1 else None

    color_img = load_or_create_image(user_path)
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # Save the original image first so it can be compared with the results.
    save("0_original_image.png", color_img)

    print("=" * 70)
    print(" FEATURE DESCRIPTOR EXTRACTION")
    print("=" * 70)

    run_canny(gray)
    run_harris(gray, color_img)
    run_sift(gray, color_img)
    run_orb(gray, color_img)
    run_lbp(gray)
    run_hog(gray)
    run_color_histogram(color_img)

    preview_path = create_preview()

    output_abspath = os.path.abspath(OUTPUT_DIR)
    print("\n[DONE] Feature extraction finished.")
    print(f"       All results are in: {output_abspath}")

    # Automatically pop up the combined preview image so the user doesn't
    # have to go hunting for the output folder themselves. Auto-open isn't
    # guaranteed to work on every OS/terminal, so always print the path too.
    if preview_path:
        print("[PREVIEW] Attempting to open the preview image automatically...")
        opened = open_path(preview_path)
        print("[PREVIEW] Also opening the output folder in File Explorer...")
        open_containing_folder(preview_path)
        if not opened:
            print("          Could not confirm the image viewer launched.")
        print(f"          If nothing opened, open this file manually:\n          {preview_path}")


if __name__ == "__main__":
    main()