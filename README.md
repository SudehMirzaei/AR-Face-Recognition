# AR-Face-Recognition

A Python project that demonstrates two fundamental computer vision tasks using OpenCV:

- **Face Detection** using OpenCV's pre-trained Haar Cascade classifiers.
- **Face Matching** using ORB (Oriented FAST and Rotated BRIEF) feature detection with Brute Force Matching.

---


# FaceDetector.py

`FaceDetector.py` detects human faces using OpenCV's pre-trained Haar Cascade classifiers.

## How It Works

### 1. Loading the Haar Cascade

The program loads OpenCV's pre-trained XML cascade files for both:

- Frontal face detection
- Profile (side-view) face detection

When the XML file is loaded, OpenCV performs the following internally:

1. Reads the XML file from disk.
2. Parses the XML structure.
3. Creates an in-memory representation of the cascade.
4. The cascade is then ready to detect faces.

---

### Cascade Classifier

The frontal face classifier contains **25 cascade stages**.

Instead of applying every test to every image region, the classifier eliminates unlikely regions as early as possible.

Example:

**Stage 1**

- Checks for basic contrast variation.
- Many non-face regions are rejected immediately.

**Stage 2**

- Looks for two dark horizontal regions that resemble the eye area.
- Additional regions are rejected.

The remaining candidate regions continue through deeper stages until either rejected or classified as a face.

This process is called a **cascade** because most image regions are discarded early, allowing only promising candidates to continue through the remaining stages.

---

### Frontal and Profile Face Detection

The detector first attempts to locate **frontal faces**.

If no face is detected, the program automatically switches to the **profile face detector**, which is designed for side-view faces where:

- Only one eye may be visible.
- The nose appears from the side.
- The face is no longer symmetrical.

This improves detection robustness when the subject is not facing the camera.

---

### Image Preprocessing

Before detection, the input image is converted from **BGR** to **grayscale**.

Face detection does not require color information, and processing a single channel is approximately **three times faster** than processing three color channels.

After conversion, histogram equalization (`equalizeHist`) is applied to enhance image contrast, improving detection under varying lighting conditions.

---

### Multi-Scale Face Detection

Faces may appear at different sizes depending on their distance from the camera.

To handle this, OpenCV builds an **image scale pyramid**, repeatedly resizing the image and searching for faces at each scale.

This allows the detector to recognize both small and large faces within the same image.

---

### Detection Output

If a face is found, the detector returns a bounding rectangle represented by:

```python
(x, y, width, height)
```

Where:

- **x** → left coordinate
- **y** → top coordinate
- **width** → width of the detected face
- **height** → height of the detected face

If no frontal face is detected, the profile face detector is executed automatically.

---

# FaceMatcher.py

`FaceMatcher.py` compares two face images using **ORB (Oriented FAST and Rotated BRIEF)** feature detection and a **Brute Force Matcher (BFMatcher)**.

Rather than comparing images pixel-by-pixel, ORB extracts distinctive local features from each image and matches them based on binary descriptors.

---

## ORB Overview

ORB combines two techniques:

- **FAST** for keypoint detection.
- **BRIEF** (with orientation compensation) for descriptor computation.

This makes ORB both fast and suitable for real-time applications.

---

## Step 1 — Keypoint Detection

For each image, ORB first detects keypoints using the **FAST (Features from Accelerated Segment Test)** algorithm.

### FAST Algorithm

For every pixel **p** with intensity **I(p)**:

1. Consider a circle of **16 surrounding pixels**.
2. Pixel **p** is classified as a corner if there exists a contiguous set of pixels that are:

- Brighter than:

```
I(p) + threshold
```

or

- Darker than:

```
I(p) - threshold
```

### High-Speed Rejection Test

FAST achieves its speed by first checking only four pixels on the circle:

- 1
- 5
- 9
- 13

If at least three of these pixels fail the brightness/darkness test, the candidate is rejected immediately without checking the remaining pixels.

---

### Harris Corner Ranking

After FAST detects candidate keypoints, ORB applies the **Harris Corner Measure** to rank them by quality and retain the strongest ones.

---

## KeyPoint Structure

Each detected keypoint contains:

| Property | Description |
|----------|-------------|
| `kp.pt` | (x, y) coordinates |
| `kp.size` | Diameter of the meaningful region |
| `kp.angle` | Orientation in degrees |
| `kp.response` | Corner strength |
| `kp.octave` | Pyramid level where detected |
| `kp.class_id` | Object class ID (typically `-1`) |

---

## Step 2 — Descriptor Computation

For every detected keypoint, ORB computes a **256-bit binary descriptor** describing the local image region surrounding that keypoint.

These binary descriptors provide a compact representation of image features while remaining invariant to rotation.

---

## Brute Force Matching

The project uses OpenCV's **Brute Force Matcher (BFMatcher)**.

BFMatcher is the simplest feature matching algorithm.

For two descriptor sets:

```
A = {a₁, a₂, ..., aₙ}

B = {b₁, b₂, ..., bₘ}
```

the matcher computes:

```
distance(aᵢ, bⱼ)
```

for every possible pair.

This creates an **n × m distance matrix**, where each element represents the similarity between one descriptor from Image A and one descriptor from Image B.

---

## Hamming Distance

Since ORB descriptors are binary, the matcher uses **Hamming Distance** (`NORM_HAMMING`).

Hamming distance counts the number of differing bits between two binary descriptors.

Examples:

- Distance **0** → Perfect match
- Larger distance → Less similar descriptors

---

## Cross Check

The matcher is configured with:

```python
crossCheck=True
```

A match is accepted only if:

- Descriptor A's best match is Descriptor B.
- Descriptor B's best match is Descriptor A.

This bidirectional verification significantly reduces false matches.

---

## Match Object

The matcher returns a list of `cv2.DMatch` objects.

Each match contains:

| Property | Description |
|----------|-------------|
| `queryIdx` | Index of descriptor in Image 1 |
| `trainIdx` | Index of descriptor in Image 2 |
| `imgIdx` | Image index (usually 0) |
| `distance` | Hamming distance between descriptors |

---

## Best Matches

The resulting matches are sorted according to Hamming distance.

The project selects the **top 50 matches**, representing the strongest correspondences between the two face images.

---

# Requirements

- Python 3.x
- OpenCV (`opencv-python`)

Install dependencies using:

```bash
pip install opencv-python
```

---

# Concepts Covered

- Haar Cascade Classifiers
- Multi-stage Cascade Detection
- Frontal and Profile Face Detection
- Histogram Equalization
- Multi-scale Image Pyramids
- ORB Feature Detection
- FAST Corner Detection
- Harris Corner Ranking
- Binary Feature Descriptors
- Brute Force Feature Matching
- Hamming Distance
- Cross-Check Matching

---

# License

This project is intended for educational purposes and demonstrates classical computer vision techniques available in OpenCV.

