"""
Main EyeTracker class - single entry point for eye tracking.

Provides a simple interface to track eye features from video frames.
"""

import cv2
import numpy as np
from typing import Optional, Iterator, Tuple, Generator

from .data import EyeData, PupilData, IrisData, CornealReflectionData, EyelidData
from .settings import EyeTrackerSettings
from .pupil_tracking import find_pupil_blob, refine_pupil_ellipse
from .cr_tracking import find_corneal_reflections
from .iris_tracking import find_iris
from .eyelid_tracking import find_eyelids


class EyeTracker:
    """
    Main eye tracking class.
    
    Provides a simple interface to track eye features from cv2.VideoCapture
    or individual frames. Tracks pupil, iris, corneal reflections, and eyelids.
    
    Example usage:
        ```python
        import cv2
        from python_openiris import EyeTracker, EyeTrackerSettings
        
        # Create tracker with custom settings
        settings = EyeTrackerSettings(dark_threshold=40)
        tracker = EyeTracker(settings)
        
        # Open video
        cap = cv2.VideoCapture("eye_video.mp4")
        
        # Track from video
        for result in tracker.track_video(cap):
            print(f"Pupil at: {result.pupil.center}")
        
        # Or track single frame
        ret, frame = cap.read()
        result = tracker.track_frame(frame)
        ```
    
    Args:
        settings: Optional EyeTrackerSettings. Uses defaults if not provided.
    """
    
    def __init__(self, settings: Optional[EyeTrackerSettings] = None):
        """
        Initialize the eye tracker.
        
        Args:
            settings: Optional tracker settings. Uses defaults if not provided.
        """
        self.settings = settings or EyeTrackerSettings()
        self._frame_number = 0
    
    def track_frame(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        timestamp: float = 0.0
    ) -> EyeData:
        """
        Track eye features in a single frame.
        
        Args:
            frame: BGR or grayscale image of the eye.
            roi: Optional region of interest (x, y, width, height).
                 If not provided, uses the entire frame.
            timestamp: Optional timestamp for this frame.
        
        Returns:
            EyeData containing all detected eye features.
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        img_h, img_w = gray.shape[:2]
        
        # Default ROI is entire image
        if roi is None:
            roi = (0, 0, img_w, img_h)
        
        # Track pupil
        pupil = find_pupil_blob(gray, roi, self.settings)
        
        if pupil.is_empty:
            self._frame_number += 1
            return EyeData(
                frame_number=self._frame_number - 1,
                timestamp=timestamp,
                tracking_success=False
            )
        
        # Refine pupil with ellipse fitting
        pupil = refine_pupil_ellipse(gray, pupil, self.settings)
        
        # Track iris (based on pupil and settings)
        iris = find_iris(pupil, self.settings)
        
        # Track corneal reflections
        corneal_reflections = []
        if self.settings.track_corneal_reflections:
            corneal_reflections = find_corneal_reflections(gray, pupil, self.settings)
        
        # Track eyelids
        eyelids = EyelidData()
        if self.settings.track_eyelids:
            eyelids = find_eyelids(gray, pupil, iris, self.settings)
        
        self._frame_number += 1
        
        return EyeData(
            pupil=pupil,
            iris=iris,
            corneal_reflections=corneal_reflections,
            eyelids=eyelids,
            timestamp=timestamp,
            frame_number=self._frame_number - 1,
            tracking_success=True
        )
    
    def track_video(
        self,
        video: cv2.VideoCapture,
        roi: Optional[Tuple[int, int, int, int]] = None,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[EyeData, np.ndarray], None, None]:
        """
        Track eye features in a video stream.
        
        Generator that yields tracking results for each frame.
        
        Args:
            video: cv2.VideoCapture object (can be file or camera).
            roi: Optional region of interest (x, y, width, height).
                 If not provided, uses the entire frame.
            max_frames: Optional maximum number of frames to process.
        
        Yields:
            Tuple of (EyeData, frame) for each processed frame.
        """
        self._frame_number = 0
        frames_processed = 0
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default FPS if not available
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Calculate timestamp
            timestamp = frames_processed / fps
            
            # Track this frame
            result = self.track_frame(frame, roi, timestamp)
            
            yield result, frame
            
            frames_processed += 1
            if max_frames is not None and frames_processed >= max_frames:
                break
    
    def reset(self):
        """Reset the tracker state (frame counter, etc.)."""
        self._frame_number = 0


def draw_eye_overlay(
    frame: np.ndarray,
    eye_data: EyeData,
    draw_pupil: bool = True,
    draw_iris: bool = True,
    draw_corneal_reflections: bool = True,
    draw_eyelids: bool = True,
    pupil_color: Tuple[int, int, int] = (0, 255, 0),
    iris_color: Tuple[int, int, int] = (255, 0, 0),
    cr_color: Tuple[int, int, int] = (0, 255, 255),
    eyelid_color: Tuple[int, int, int] = (255, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw eye tracking overlay on a frame.
    
    Args:
        frame: BGR image to draw on (will be modified in place).
        eye_data: Tracking results to visualize.
        draw_pupil: Whether to draw the pupil ellipse.
        draw_iris: Whether to draw the iris circle.
        draw_corneal_reflections: Whether to draw corneal reflections.
        draw_eyelids: Whether to draw eyelid points.
        pupil_color: BGR color for pupil (default: green).
        iris_color: BGR color for iris (default: blue).
        cr_color: BGR color for corneal reflections (default: cyan).
        eyelid_color: BGR color for eyelids (default: magenta).
        thickness: Line thickness for drawing.
    
    Returns:
        The frame with overlays drawn.
    """
    if not eye_data.tracking_success:
        return frame
    
    # Ensure frame is BGR
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Draw iris (circle)
    if draw_iris and not eye_data.iris.is_empty:
        center = (int(eye_data.iris.center[0]), int(eye_data.iris.center[1]))
        radius = int(eye_data.iris.radius)
        cv2.circle(frame, center, radius, iris_color, thickness)
    
    # Draw pupil (ellipse)
    if draw_pupil and not eye_data.pupil.is_empty:
        center = (int(eye_data.pupil.center[0]), int(eye_data.pupil.center[1]))
        axes = (int(eye_data.pupil.size[0] / 2), int(eye_data.pupil.size[1] / 2))
        angle = eye_data.pupil.angle
        cv2.ellipse(frame, center, axes, angle, 0, 360, pupil_color, thickness)
        # Draw center point
        cv2.circle(frame, center, 3, pupil_color, -1)
    
    # Draw corneal reflections
    if draw_corneal_reflections:
        for cr in eye_data.corneal_reflections:
            if not cr.is_empty:
                center = (int(cr.center[0]), int(cr.center[1]))
                radius = int(max(cr.size[0], cr.size[1]) / 2)
                cv2.circle(frame, center, max(radius, 3), cr_color, thickness)
    
    # Draw eyelids
    if draw_eyelids and not eye_data.eyelids.is_empty:
        # Draw upper eyelid points
        for i, point in enumerate(eye_data.eyelids.upper):
            if point != (0.0, 0.0):
                cv2.circle(frame, (int(point[0]), int(point[1])), 4, eyelid_color, -1)
                if i > 0:
                    prev = eye_data.eyelids.upper[i - 1]
                    if prev != (0.0, 0.0):
                        cv2.line(
                            frame,
                            (int(prev[0]), int(prev[1])),
                            (int(point[0]), int(point[1])),
                            eyelid_color,
                            thickness
                        )
        
        # Draw lower eyelid points
        for i, point in enumerate(eye_data.eyelids.lower):
            if point != (0.0, 0.0):
                cv2.circle(frame, (int(point[0]), int(point[1])), 4, eyelid_color, -1)
                if i > 0:
                    prev = eye_data.eyelids.lower[i - 1]
                    if prev != (0.0, 0.0):
                        cv2.line(
                            frame,
                            (int(prev[0]), int(prev[1])),
                            (int(point[0]), int(point[1])),
                            eyelid_color,
                            thickness
                        )
    
    return frame
