# OpenIris Python - Eye Tracking with OpenCV

A Python implementation of the OpenIris eye tracking algorithms using OpenCV. This package provides real-time tracking of:

- **Pupil** - Center position, size, and orientation
- **Iris** - Center and radius
- **Corneal Reflections** - Multiple bright spots from light sources
- **Eyelids** - Upper and lower eyelid contour points

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/ocular-motor-lab/OpenIris.git
cd OpenIris

# Install dependencies
pip install opencv-python numpy

# Or install as a package
pip install -e .
```

### Requirements

- Python 3.8+
- OpenCV 4.5+
- NumPy 1.20+

## Quick Start

### Basic Usage

```python
import cv2
from python_openiris import EyeTracker, EyeTrackerSettings

# Create tracker with default settings
tracker = EyeTracker()

# Or customize settings
settings = EyeTrackerSettings(
    dark_threshold=40,      # Lower = detect darker pupils
    bright_threshold=200,   # Higher = detect brighter reflections
    iris_radius=80.0,       # Expected iris radius in pixels
)
tracker = EyeTracker(settings)

# Open video file or camera
cap = cv2.VideoCapture("eye_video.mp4")
# Or use webcam: cap = cv2.VideoCapture(0)

# Track frames
for eye_data, frame in tracker.track_video(cap):
    if eye_data.tracking_success:
        print(f"Pupil at: {eye_data.pupil.center}")
        print(f"Corneal reflections: {len(eye_data.corneal_reflections)}")

cap.release()
```

### Single Frame Processing

```python
import cv2
from python_openiris import EyeTracker

tracker = EyeTracker()

# Read a single frame
frame = cv2.imread("eye_image.png")

# Track eye features
result = tracker.track_frame(frame)

if result.tracking_success:
    print(f"Pupil center: {result.pupil.center}")
    print(f"Pupil size: {result.pupil.size}")
    print(f"Iris radius: {result.iris.radius}")
```

### Drawing Overlays

```python
import cv2
from python_openiris import EyeTracker
from python_openiris.tracker import draw_eye_overlay

tracker = EyeTracker()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = tracker.track_frame(frame)
    
    # Draw detected features on frame
    frame_with_overlay = draw_eye_overlay(frame, result)
    
    cv2.imshow("Eye Tracking", frame_with_overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Sample Script

A complete demo script is provided:

```bash
# Run with video file
python sample_eye_tracker.py path/to/eye_video.mp4

# Run with webcam
python sample_eye_tracker.py

# Customize thresholds
python sample_eye_tracker.py video.mp4 --dark-threshold 40 --bright-threshold 180

# Enable eyelid tracking
python sample_eye_tracker.py video.mp4 --track-eyelids

# Save output video
python sample_eye_tracker.py video.mp4 --output tracked_output.mp4
```

### Controls in Sample Script

- `q` - Quit
- `p` - Pause/Resume
- `s` - Save current frame
- `+`/`-` - Adjust dark threshold (pupil detection)
- `[`/`]` - Adjust bright threshold (corneal reflection detection)

## API Reference

### EyeTrackerSettings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dark_threshold` | int | 50 | Threshold for pupil detection (0-255) |
| `bright_threshold` | int | 200 | Threshold for corneal reflection detection (0-255) |
| `min_pupil_radius` | float | 10.0 | Minimum pupil radius in pixels |
| `max_pupil_radius` | float | 80.0 | Maximum pupil radius in pixels |
| `min_cr_radius` | float | 2.0 | Minimum corneal reflection radius |
| `max_cr_radius` | float | 20.0 | Maximum corneal reflection radius |
| `iris_radius` | float | 80.0 | Expected iris radius in pixels |
| `track_corneal_reflections` | bool | True | Enable CR tracking |
| `track_eyelids` | bool | False | Enable eyelid tracking |
| `eyelid_tracking_method` | str | "fixed" | Method: "none", "fixed", or "hough" |

### EyeData (Tracking Result)

| Attribute | Type | Description |
|-----------|------|-------------|
| `pupil` | PupilData | Pupil ellipse (center, size, angle) |
| `iris` | IrisData | Iris circle (center, radius) |
| `corneal_reflections` | List[CornealReflectionData] | Detected reflections (up to 5) |
| `eyelids` | EyelidData | Upper/lower eyelid points |
| `timestamp` | float | Frame timestamp in seconds |
| `frame_number` | int | Frame number |
| `tracking_success` | bool | Whether tracking succeeded |

### PupilData

| Attribute | Type | Description |
|-----------|------|-------------|
| `center` | Tuple[float, float] | (x, y) center position |
| `size` | Tuple[float, float] | (width, height) of ellipse |
| `angle` | float | Rotation angle in degrees |

## Algorithm Overview

The tracking pipeline follows the original OpenIris C# implementation:

1. **Pupil Detection (Blob)**: 
   - Threshold image to find dark pixels
   - Apply morphological operations (opening/closing)
   - Find contours and score blobs by compactness
   - Select best blob as pupil candidate

2. **Pupil Refinement (Ellipse Fitting)**:
   - Extract region around detected pupil
   - Find contour of thresholded pupil
   - Fit ellipse to contour for precise measurements

3. **Corneal Reflection Detection**:
   - Search within iris region for bright spots
   - Filter by size and distance from pupil
   - Sort by distance (closest first)

4. **Eyelid Detection**:
   - Fixed method: Place points at fixed distances from pupil
   - Hough method: Use edge detection and Hough lines

## Credits

Based on the original OpenIris framework:

> Roksana Sadeghi, Ryan Ressmeyer, Jacob Yates, and Jorge Otero-Millan. 2024. 
> Open Iris - An Open Source Framework for Video-Based Eye-Tracking Research and Development. 
> In Proceedings of the 2024 Symposium on Eye Tracking Research and Applications (ETRA '24).

Original C# implementation: [OpenIris on GitHub](https://github.com/ocular-motor-lab/OpenIris)

## License

This project is licensed under the LGPL-3.0 License - see the LICENSE file for details.
