"""
Data structures for eye tracking results.

Based on the OpenIris C# data structures.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np


@dataclass
class PupilData:
    """
    Data structure containing the geometric properties of the measured pupil.
    
    Attributes:
        center: The (x, y) center of the pupil ellipse.
        size: The (width, height) size of the pupil ellipse.
        angle: The angle between the horizontal axis and the width in degrees.
               Positive value means counter-clockwise rotation.
    """
    center: Tuple[float, float] = (0.0, 0.0)
    size: Tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0
    
    @property
    def is_empty(self) -> bool:
        """Returns True if the pupil information is empty."""
        return self.size[0] == 0.0 and self.size[1] == 0.0
    
    @property
    def radius(self) -> float:
        """Returns the average radius of the pupil."""
        return (self.size[0] + self.size[1]) / 4.0


@dataclass
class IrisData:
    """
    Data structure containing the geometric properties of the measured iris.
    
    Attributes:
        center: The (x, y) center of the iris circle.
        radius: The radius of the iris circle.
    """
    center: Tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0
    
    @property
    def is_empty(self) -> bool:
        """Returns True if the iris information is empty."""
        return self.radius == 0.0


@dataclass
class CornealReflectionData:
    """
    Data structure containing the geometric properties of a measured corneal reflection.
    
    Attributes:
        center: The (x, y) center of the reflection.
        size: The (width, height) size of the reflection.
        angle: The angle of the reflection ellipse in degrees.
    """
    center: Tuple[float, float] = (0.0, 0.0)
    size: Tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0
    
    @property
    def is_empty(self) -> bool:
        """Returns True if the corneal reflection information is empty."""
        return self.size[0] == 0.0 and self.size[1] == 0.0


@dataclass
class EyelidData:
    """
    Data structure containing the data about the eyelids.
    
    The eyelids are represented as 4 points each for upper and lower lids.
    Points go from left to right.
    
    Attributes:
        upper: Array of 4 (x, y) points for the upper eyelid contour.
        lower: Array of 4 (x, y) points for the lower eyelid contour.
    """
    upper: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0)] * 4
    )
    lower: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0)] * 4
    )
    
    @property
    def is_empty(self) -> bool:
        """Returns True if the eyelid information is empty."""
        return all(p == (0.0, 0.0) for p in self.upper + self.lower)


@dataclass
class EyeData:
    """
    Complete eye tracking data for a single frame.
    
    Attributes:
        pupil: Pupil tracking data.
        iris: Iris tracking data.
        corneal_reflections: List of detected corneal reflections (up to 5).
        eyelids: Eyelid tracking data.
        timestamp: Frame timestamp in seconds (if available).
        frame_number: Frame number in the video.
        tracking_success: Whether tracking was successful for this frame.
    """
    pupil: PupilData = field(default_factory=PupilData)
    iris: IrisData = field(default_factory=IrisData)
    corneal_reflections: List[CornealReflectionData] = field(default_factory=list)
    eyelids: EyelidData = field(default_factory=EyelidData)
    timestamp: float = 0.0
    frame_number: int = 0
    tracking_success: bool = False
