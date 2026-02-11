"""
Corneal reflection tracking using blob detection.

Based on CornealReflectionTracking.cs from the original C# implementation.
"""

import cv2
import numpy as np
from typing import List, Tuple
from .data import PupilData, CornealReflectionData
from .settings import EyeTrackerSettings


def find_corneal_reflections(
    image: np.ndarray,
    pupil: PupilData,
    settings: EyeTrackerSettings
) -> List[CornealReflectionData]:
    """
    Find corneal reflections (bright spots) near the pupil.
    
    Uses blob detection on bright pixels, filtering by size and distance to pupil.
    
    Args:
        image: Grayscale image of the eye.
        pupil: Detected pupil data.
        settings: Eye tracker settings.
    
    Returns:
        List of CornealReflectionData, sorted by distance to pupil (closest first).
    """
    if pupil.is_empty:
        return []
    
    # Parameters
    iris_radius = settings.iris_radius
    threshold = settings.bright_threshold
    min_blob_area = np.pi * (settings.min_cr_radius ** 2)
    max_blob_area = np.pi * (settings.max_cr_radius ** 2)
    blur_size = max(1, int(np.ceil(settings.min_cr_radius / 2)))
    
    img_h, img_w = image.shape[:2]
    
    # Define ROI around iris (using iris radius for the ROI)
    square_size = int(iris_radius * 2)
    iris_roi_x = int(pupil.center[0] - square_size / 2)
    iris_roi_y = int(pupil.center[1] - square_size / 2)
    
    # Clamp to image bounds
    iris_roi_x = max(0, min(iris_roi_x, img_w - 1))
    iris_roi_y = max(0, min(iris_roi_y, img_h - 1))
    iris_roi_w = min(square_size, img_w - iris_roi_x)
    iris_roi_h = min(square_size, img_h - iris_roi_y)
    
    if iris_roi_w <= 0 or iris_roi_h <= 0:
        return []
    
    # Extract and preprocess ROI
    roi_image = image[iris_roi_y:iris_roi_y+iris_roi_h, iris_roi_x:iris_roi_x+iris_roi_w]
    
    # Blur to reduce noise
    smooth_blur_size = blur_size + 1
    if smooth_blur_size % 2 == 0:
        smooth_blur_size += 1
    img_blurred = cv2.blur(roi_image, (smooth_blur_size, smooth_blur_size))
    
    # Threshold to find bright pixels (corneal reflections)
    _, img_binary = cv2.threshold(img_blurred, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    if blur_size > 0:
        kernel_size = max(1, blur_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Opening: remove small bright spots
        img_binary = cv2.erode(img_binary, kernel, iterations=1)
        img_binary = cv2.dilate(img_binary, kernel, iterations=1)
        
        # Closing: fill gaps in reflections
        img_binary = cv2.dilate(img_binary, kernel, iterations=1)
        img_binary = cv2.erode(img_binary, kernel, iterations=1)
    
    # Find contours (blobs)
    contours, _ = cv2.findContours(
        img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Process each blob
    corneal_reflections = []
    pupil_center_local = (
        pupil.center[0] - iris_roi_x,
        pupil.center[1] - iris_roi_y
    )
    local_iris_radius = iris_radius
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_blob_area or area > max_blob_area:
            continue
        
        # Get bounding box and centroid
        bx, by, bw, bh = cv2.boundingRect(contour)
        
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx = bx + bw / 2
            cy = by + bh / 2
        
        # Calculate distance to pupil center
        dist_to_pupil = np.sqrt(
            (cx - pupil_center_local[0]) ** 2 + 
            (cy - pupil_center_local[1]) ** 2
        )
        
        # Skip if too far from pupil
        if dist_to_pupil > local_iris_radius:
            continue
        
        # Score based on compactness (circle within bounding box = pi/4 ≈ 0.78)
        box_area = max(bw, bh) ** 2
        if box_area == 0:
            continue
        score = area / box_area
        
        if score > 0.3:
            # Convert to original image coordinates
            cr = CornealReflectionData(
                center=(cx + iris_roi_x, cy + iris_roi_y),
                size=(float(bw), float(bh)),
                angle=90.0
            )
            corneal_reflections.append(cr)
    
    # Sort by distance to pupil (closest first)
    corneal_reflections.sort(
        key=lambda cr: np.sqrt(
            (pupil.center[0] - cr.center[0]) ** 2 +
            (pupil.center[1] - cr.center[1]) ** 2
        )
    )
    
    # Return at most 5 corneal reflections
    return corneal_reflections[:5]
