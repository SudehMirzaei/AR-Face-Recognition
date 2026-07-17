# AR Face Recognition

A Python-based face detection and recognition system using OpenCV.

## Overview

This repository contains two main modules for face recognition tasks:

- FaceDetector.py – Detects faces in images using Haar Cascade classifiers
- FaceMatcher.py – Matches detected faces against known identities

---

## FaceDetector.py

### Description
A robust face detection module that uses OpenCV's pre-trained Haar Cascade classifiers to detect both frontal and profile faces in images.

### How It Works

#### Cascade Classifiers
The detector loads two pre-trained XML classifiers:

1. Frontal Face Cascade – Detects faces looking directly at the camera
2. Profile Face Cascade – Detects faces in side view (when frontal detection fails)

#### Internal Process
When a cascade classifier is loaded:

- OpenCV reads the XML file from disk
- It parses the XML structure
- Creates an in-memory representation of the cascade
- The cascade becomes ready for object detection

#### The Cascade Elimination Process
The frontal face cascade consists of 25 stages that work like a waterfall (hence the name "cascade"):

| Stage    | Check Performed                              | Result                          |
|----------|----------------------------------------------|---------------------------------|
| Stage 1  | Quick check – Is there any contrast variation? | Some objects rejected          |
| Stage 2  | Are there 2 dark horizontal regions?         | More candidates eliminated      |
| Stages 3-25 | Increasingly complex pattern matching     | Only promising candidates pass  |

This cascading approach is highly efficient because most candidates are eliminated in the early, simple stages. Only regions that might actually contain faces "cascade down" to the deeper, more computationally expensive stages.

#### Profile Face Detection
When the frontal face detector cannot find any faces (person is in side view, only one eye visible, nose on the side, face appears asymmetric), the module automatically falls back to the profile face detector to capture side-profile faces.

### Detection Pipeline

1. Convert to Grayscale – Color image is converted to grayscale because:
   - Face detection doesn't require color information
   - Processing 1 channel is ~3× faster than processing 3 channels (OpenCV default is BGR)

2. Histogram Equalization – Applies equalizeHist() to enhance image contrast for better detection accuracy

3. Multi-Scale Detection – Creates a scale pyramid by resizing the image multiple times, simulating looking at the photo from different distances

4. Bounding Box Output – Each detected face returns a tuple of 4 elements: (x, y, width, height)
