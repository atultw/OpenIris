"""
Iris tracking.

Based on IrisTracker.cs from the original C# implementation.
Currently uses a fixed radius approach based on settings.
"""

from .data import PupilData, IrisData
from .settings import EyeTrackerSettings


def find_iris(pupil: PupilData, settings: EyeTrackerSettings) -> IrisData:
    """
    Find the iris based on the pupil position.
    
    Currently uses the configured iris radius setting.
    The center is assumed to be at the pupil center.
    
    Args:
        pupil: Detected pupil data.
        settings: Eye tracker settings.
    
    Returns:
        IrisData with the iris information.
    """
    if pupil.is_empty:
        return IrisData()
    
    return IrisData(
        center=pupil.center,
        radius=settings.iris_radius
    )
