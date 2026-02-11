"""
Settings for eye tracking.

Based on the EyeTrackingPipelineJOMSettings from the original C# implementation.
"""

from dataclasses import dataclass


@dataclass
class EyeTrackerSettings:
    """
    Configuration settings for the eye tracker.
    
    Attributes:
        dark_threshold: Threshold for detecting dark pixels (pupil). Range: 0-255.
        bright_threshold: Threshold for detecting bright pixels (corneal reflections). Range: 0-255.
        min_pupil_radius: Minimum pupil radius in pixels.
        max_pupil_radius: Maximum pupil radius in pixels.
        min_cr_radius: Minimum corneal reflection radius in pixels.
        max_cr_radius: Maximum corneal reflection radius in pixels.
        iris_radius: Expected iris radius in pixels.
        track_corneal_reflections: Whether to track corneal reflections.
        track_eyelids: Whether to track eyelids.
        eyelid_tracking_method: Method for eyelid tracking ('none', 'fixed', 'hough').
    """
    # Threshold settings
    dark_threshold: int = 50
    bright_threshold: int = 200
    
    # Pupil size constraints (in pixels)
    min_pupil_radius: float = 10.0
    max_pupil_radius: float = 80.0
    
    # Corneal reflection size constraints (in pixels)
    min_cr_radius: float = 2.0
    max_cr_radius: float = 20.0
    
    # Iris settings
    iris_radius: float = 80.0
    
    # Feature tracking toggles
    track_corneal_reflections: bool = True
    track_eyelids: bool = False
    eyelid_tracking_method: str = "fixed"  # 'none', 'fixed', or 'hough'
    
    # Processing settings
    processing_image_size: int = 200  # Size for internal processing
