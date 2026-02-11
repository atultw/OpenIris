"""
Eyelid tracking.

Based on EyeLidTracking.cs from the original C# implementation.
Provides fixed and Hough-line based eyelid detection.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from .data import PupilData, IrisData, EyelidData
from .settings import EyeTrackerSettings


def find_eyelids(
    image: np.ndarray,
    pupil: PupilData,
    iris: IrisData,
    settings: EyeTrackerSettings
) -> EyelidData:
    """
    Find the eyelids using the configured method.
    
    Args:
        image: Grayscale image of the eye.
        pupil: Detected pupil data.
        iris: Detected iris data.
        settings: Eye tracker settings.
    
    Returns:
        EyelidData with upper and lower eyelid points.
    """
    if pupil.is_empty:
        return EyelidData()
    
    method = settings.eyelid_tracking_method.lower()
    
    if method == "none":
        return EyelidData()
    elif method == "fixed":
        return find_eyelids_fixed(pupil, iris)
    elif method == "hough":
        return find_eyelids_hough(image, pupil, iris, settings)
    else:
        return EyelidData()


def find_eyelids_fixed(pupil: PupilData, iris: IrisData) -> EyelidData:
    """
    Find eyelids using fixed positions relative to the pupil.
    
    Simple method that places eyelid points at fixed distances from the pupil.
    
    Args:
        pupil: Detected pupil data.
        iris: Detected iris data.
    
    Returns:
        EyelidData with fixed eyelid positions.
    """
    pupil_radius = (pupil.size[0] + pupil.size[1]) / 4.0
    iris_radius = iris.radius if iris.radius > 0 else pupil_radius * 2
    
    y_top = pupil.center[1] - pupil_radius
    y_bottom = pupil.center[1] + pupil_radius
    
    # Define 4 points for each eyelid (left to right)
    upper = [
        (pupil.center[0] - iris_radius * 0.75, y_top),
        (pupil.center[0] - iris_radius * 0.5, y_top),
        (pupil.center[0] + iris_radius * 0.5, y_top),
        (pupil.center[0] + iris_radius * 0.75, y_top),
    ]
    
    lower = [
        (pupil.center[0] - iris_radius * 0.75, y_bottom),
        (pupil.center[0] - iris_radius * 0.5, y_bottom),
        (pupil.center[0] + iris_radius * 0.5, y_bottom),
        (pupil.center[0] + iris_radius * 0.75, y_bottom),
    ]
    
    return EyelidData(upper=upper, lower=lower)


def find_eyelids_hough(
    image: np.ndarray,
    pupil: PupilData,
    iris: IrisData,
    settings: EyeTrackerSettings
) -> EyelidData:
    """
    Find eyelids using Hough line detection.
    
    Based on FindEyelidsHoughLines from the original C# implementation.
    Finds four segments of the eyelids independently around the pupil.
    
    Args:
        image: Grayscale image of the eye.
        pupil: Detected pupil data.
        iris: Detected iris data.
        settings: Eye tracker settings.
    
    Returns:
        EyelidData with detected eyelid positions.
    """
    # Parameters
    pixels_per_eye_radius = 80
    hough_threshold = 20
    hough_rho_resolution = 2
    hough_theta_resolution = 5  # degrees
    
    iris_radius = iris.radius if iris.radius > 0 else settings.iris_radius
    eye_globe_radius = iris_radius * 2  # Approximate eyeglobe from iris
    
    img_h, img_w = image.shape[:2]
    
    # Calculate scale for processing
    scale = eye_globe_radius / pixels_per_eye_radius
    if scale == 0:
        scale = 1
    
    small_w = int(round(img_w / scale))
    small_h = int(round(img_h / scale))
    scale_x = small_w / img_w
    scale_y = small_h / img_h
    
    # Resize image
    image_resize = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_CUBIC)
    
    # Scaled parameters
    iris_radius_scaled = iris_radius * scale_x
    pupil_center_scaled = (pupil.center[0] * scale_x, pupil.center[1] * scale_y)
    eye_globe_center_scaled = pupil_center_scaled  # Approximate
    
    # Search area dimensions
    max_area_width = int(iris_radius_scaled * 0.8)
    center_gap = int(iris_radius_scaled * 0.3)
    
    # Vertical positions
    top_y = int(max(
        pupil_center_scaled[1] - iris_radius_scaled * 1.1,
        eye_globe_center_scaled[1] - iris_radius_scaled * 1.1
    ))
    top_height = int(iris_radius_scaled * 1.0)
    bottom_y = int(eye_globe_center_scaled[1] + iris_radius_scaled / 2.0)
    bottom_height = int(iris_radius_scaled * 1.3)
    
    # Horizontal positions
    x_left = int(max(0, pupil_center_scaled[0] - max_area_width - center_gap))
    x_right = int(min(small_w - max_area_width, pupil_center_scaled[0] + center_gap))
    
    # Create default eyelid lines
    def create_default_line(roi: Tuple[int, int, int, int], corner: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        rx, ry, rw, rh = roi
        if corner == "top_left":
            return ((rx, ry + rh), (rx + rw, ry + rh / 2))
        elif corner == "top_right":
            return ((rx, ry + rh / 2), (rx + rw, ry + rh))
        elif corner == "bottom_left":
            return ((rx, ry), (rx + rw, ry + rh / 2))
        else:  # bottom_right
            return ((rx, ry + rh / 2), (rx + rw, ry))
    
    # Find eyelid segments for each corner
    def find_segment(roi: Tuple[int, int, int, int], corner: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        rx, ry, rw, rh = roi
        
        # Clamp ROI
        rx = max(0, min(rx, small_w - 1))
        ry = max(0, min(ry, small_h - 1))
        rw = min(rw, small_w - rx)
        rh = min(rh, small_h - ry)
        
        if rw < 2 or rh < 2:
            return create_default_line((rx, ry, rw, rh), corner)
        
        # Extract ROI
        roi_img = image_resize[ry:ry+rh, rx:rx+rw]
        
        # Edge detection
        roi_img = cv2.blur(roi_img, (4, 4))
        sob_y = cv2.Sobel(roi_img, cv2.CV_64F, 0, 2, ksize=5)
        sob_x = cv2.Sobel(roi_img, cv2.CV_64F, 2, 0, ksize=5)
        
        # Combine based on corner angle range
        angle_ranges = {
            "top_left": (-0.2, 0.7),
            "top_right": (-0.7, 0.2),
            "bottom_left": (-0.4, 0.2),
            "bottom_right": (-0.2, 0.4),
        }
        cos_range = angle_ranges[corner]
        angle = np.arcsin((cos_range[0] + cos_range[1]) / 2)
        
        combined = np.abs(sob_y * np.cos(angle) + sob_x * np.sin(angle))
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        
        # Equalize and threshold
        combined = cv2.equalizeHist(combined)
        _, binary = cv2.threshold(combined, 230, 255, cv2.THRESH_BINARY)
        
        # Hough lines
        lines = cv2.HoughLines(
            binary,
            hough_rho_resolution,
            np.radians(hough_theta_resolution),
            hough_threshold
        )
        
        if lines is None:
            return create_default_line((rx, ry, rw, rh), corner)
        
        # Find best line based on corner type
        best_line = None
        best_y = 0 if corner.startswith("top") else rh
        
        for line in lines[:30]:
            rho, theta = line[0]
            
            # Check angle constraint
            cos_theta = np.cos(theta)
            if not (cos_range[0] <= cos_theta <= cos_range[1]):
                continue
            if np.sin(theta) <= 0.1:
                continue
            
            # Convert to line endpoints
            y1 = rho / np.sin(theta)
            y2 = -(np.cos(theta) / np.sin(theta)) * rw + y1
            
            # Select based on corner
            if corner == "top_left" and y2 > best_y:
                best_y = y2
                best_line = ((rx, ry + y1), (rx + rw, ry + y2))
            elif corner == "top_right" and y1 > best_y:
                best_y = y1
                best_line = ((rx, ry + y1), (rx + rw, ry + y2))
            elif corner == "bottom_left" and y1 < best_y:
                best_y = y1
                best_line = ((rx, ry + y1), (rx + rw, ry + y2))
            elif corner == "bottom_right" and y2 < best_y:
                best_y = y2
                best_line = ((rx, ry + y1), (rx + rw, ry + y2))
        
        if best_line is None:
            return create_default_line((rx, ry, rw, rh), corner)
        
        return best_line
    
    # Define ROIs for each corner
    bottom_left_roi = (x_left, bottom_y, max_area_width, bottom_height)
    bottom_right_roi = (x_right, bottom_y, max_area_width, bottom_height)
    top_left_roi = (x_left, top_y, max_area_width, top_height)
    top_right_roi = (x_right, top_y, max_area_width, top_height)
    
    # Find segments
    line_left_top = find_segment(top_left_roi, "top_left")
    line_right_top = find_segment(top_right_roi, "top_right")
    line_left_bottom = find_segment(bottom_left_roi, "bottom_left")
    line_right_bottom = find_segment(bottom_right_roi, "bottom_right")
    
    # Scale back to original coordinates
    def scale_point(p: Tuple[float, float]) -> Tuple[float, float]:
        return (p[0] / scale_x, p[1] / scale_y)
    
    upper = [
        scale_point(line_left_top[0]),
        scale_point(line_left_top[1]),
        scale_point(line_right_top[0]),
        scale_point(line_right_top[1]),
    ]
    
    lower = [
        scale_point(line_left_bottom[0]),
        scale_point(line_left_bottom[1]),
        scale_point(line_right_bottom[0]),
        scale_point(line_right_bottom[1]),
    ]
    
    return EyelidData(upper=upper, lower=lower)
