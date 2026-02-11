"""
OpenIris Python - Eye Tracking with OpenCV

A Python implementation of the OpenIris eye tracking algorithms.
Tracks pupil, iris, corneal reflections, and eyelids from close-up eye video.

Original C# implementation: https://github.com/ocular-motor-lab/OpenIris
Copyright (c) 2014-2023 Jorge Otero-Millan, Johns Hopkins University, University of California, Berkeley.
"""

from .data import (
    PupilData,
    IrisData,
    CornealReflectionData,
    EyelidData,
    EyeData,
)
from .tracker import EyeTracker
from .settings import EyeTrackerSettings

__version__ = "1.0.0"
__all__ = [
    "EyeTracker",
    "EyeTrackerSettings",
    "PupilData",
    "IrisData", 
    "CornealReflectionData",
    "EyelidData",
    "EyeData",
]
