"""
Pupil tracking using blob detection.

Based on the PupilTracking.cs from the original C# implementation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .data import PupilData
from .settings import EyeTrackerSettings


def find_pupil_blob(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    settings: EyeTrackerSettings
) -> PupilData:
    """
    Find the pupil using blob detection.
    
    The blobs are scored according to size and compactness.
    
    Args:
        image: Grayscale image of the eye.
        roi: Region of interest (x, y, width, height) containing the pupil.
        settings: Eye tracker settings.
    
    Returns:
        PupilData with the detected pupil information.
    """
    x, y, w, h = roi
    
    # Clamp ROI to image bounds
    img_h, img_w = image.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    if w < 10 or h < 10:
        return PupilData()
    
    # Parameters
    max_pup_rad = settings.max_pupil_radius
    min_pup_area = np.pi * (settings.min_pupil_radius ** 2)
    threshold_dark = settings.dark_threshold
    image_size = settings.processing_image_size
    
    # Reduce the image to increase processing speed
    # Precision at this point is not very important
    small_size = (image_size, int(round(h / w * image_size)))
    scale_down_x = small_size[0] / w
    scale_down_y = small_size[1] / h
    
    # Extract ROI and resize
    roi_image = image[y:y+h, x:x+w]
    if roi_image.size == 0:
        return PupilData()
        
    small_image = cv2.resize(roi_image, small_size, interpolation=cv2.INTER_LINEAR)
    
    # Thresholding - get binary image with dark pixels
    _, image_threshold = cv2.threshold(
        small_image, threshold_dark, 255, cv2.THRESH_BINARY_INV
    )
    
    # Morphological operations to optimize blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Opening: removes small white spots (noise)
    image_threshold = cv2.erode(image_threshold, kernel, iterations=1)
    image_threshold = cv2.dilate(image_threshold, kernel, iterations=1)
    
    # Closing: fills small black spots within blobs
    image_threshold = cv2.dilate(image_threshold, kernel, iterations=1)
    image_threshold = cv2.erode(image_threshold, kernel, iterations=1)
    
    # Find contours (blobs)
    contours, _ = cv2.findContours(
        image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Find the best blob (most likely to be the pupil)
    max_score = 0.0
    best_pupil = PupilData()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip if area is too small
        if area < min_pup_area * scale_down_x:
            continue
        
        # Get bounding box
        bx, by, bw, bh = cv2.boundingRect(contour)
        
        # Skip if bounding box is too large
        if max(bw, bh) > (max_pup_rad * 2 * scale_down_x):
            continue
        
        # Score is the ratio between area and bounding box area (compactness)
        box_area = bw * bh
        if box_area == 0:
            continue
        score = area / box_area
        
        # Lower score if touching edges (likely incomplete pupil)
        if bx < 2 or by < 2:
            score /= 2
        if (small_size[0] - (bx + bw)) < 2:
            score /= 2
        if (small_size[1] - (by + bh)) < 2:
            score /= 2
        
        if score <= max_score or score <= 0.1:
            continue
        
        # Calculate centroid
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx = bx + bw / 2
            cy = by + bh / 2
        
        max_score = score
        best_pupil = PupilData(
            center=(cx, cy),
            size=(float(bw), float(bh)),
            angle=90.0
        )
    
    # Scale back to original image coordinates
    if not best_pupil.is_empty:
        best_pupil = PupilData(
            center=(
                best_pupil.center[0] / scale_down_x + x,
                best_pupil.center[1] / scale_down_y + y
            ),
            size=(
                best_pupil.size[0] / scale_down_x,
                best_pupil.size[1] / scale_down_y
            ),
            angle=90.0
        )
    
    return best_pupil


def find_pupil_centroid(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    settings: EyeTrackerSettings
) -> PupilData:
    """
    Find the pupil using centroid of thresholded image.
    
    Simpler but less accurate than blob detection.
    
    Args:
        image: Grayscale image of the eye.
        roi: Region of interest (x, y, width, height) containing the pupil.
        settings: Eye tracker settings.
    
    Returns:
        PupilData with the detected pupil information.
    """
    x, y, w, h = roi
    
    # Clamp ROI to image bounds
    img_h, img_w = image.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    if w < 10 or h < 10:
        return PupilData()
    
    threshold_dark = settings.dark_threshold
    
    # Extract ROI
    roi_image = image[y:y+h, x:x+w]
    
    # Threshold to find dark parts (pupil)
    _, image_threshold = cv2.threshold(
        roi_image, threshold_dark, 255, cv2.THRESH_BINARY_INV
    )
    
    # Calculate moments to get centroid
    moments = cv2.moments(image_threshold, binaryImage=True)
    
    if moments["m00"] == 0:
        return PupilData()
    
    cx = moments["m10"] / moments["m00"] + x
    cy = moments["m01"] / moments["m00"] + y
    
    # Estimate radius from the area of thresholded pixels
    avg_intensity = np.mean(image_threshold) / 255.0
    radius = np.sqrt(avg_intensity * w * h / np.pi)
    
    return PupilData(
        center=(cx, cy),
        size=(radius * 2, radius * 2),
        angle=0.0
    )


def refine_pupil_ellipse(
    image: np.ndarray,
    pupil_approx: PupilData,
    settings: EyeTrackerSettings
) -> PupilData:
    """
    Refine the pupil position by fitting an ellipse to the pupil contour.
    
    Based on PositionTrackerEllipseFitting.cs from the original implementation.
    
    Args:
        image: Grayscale image of the eye.
        pupil_approx: Approximate pupil position from blob detection.
        settings: Eye tracker settings.
    
    Returns:
        Refined PupilData with ellipse fitting.
    """
    if pupil_approx.is_empty:
        return pupil_approx
    
    # Define ROI around the approximate pupil
    margin = 1.5
    cx, cy = pupil_approx.center
    pw, ph = pupil_approx.size
    
    # Calculate ROI with margin
    roi_w = int(pw * margin)
    roi_h = int(ph * margin)
    roi_x = int(cx - roi_w / 2)
    roi_y = int(cy - roi_h / 2)
    
    # Clamp to image bounds
    img_h, img_w = image.shape[:2]
    roi_x = max(0, min(roi_x, img_w - 1))
    roi_y = max(0, min(roi_y, img_h - 1))
    roi_w = min(roi_w, img_w - roi_x)
    roi_h = min(roi_h, img_h - roi_y)
    
    if roi_w < 20 or roi_h < 20:
        return pupil_approx
    
    # Extract ROI
    roi_image = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    # Threshold
    threshold_dark = settings.dark_threshold
    _, binary = cv2.threshold(roi_image, threshold_dark, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return pupil_approx
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Need at least 5 points to fit an ellipse
    if len(largest_contour) < 5:
        return pupil_approx
    
    # Fit ellipse
    try:
        ellipse = cv2.fitEllipse(largest_contour)
        center, size, angle = ellipse
        
        # Convert back to original image coordinates
        return PupilData(
            center=(center[0] + roi_x, center[1] + roi_y),
            size=size,
            angle=angle
        )
    except cv2.error:
        return pupil_approx
