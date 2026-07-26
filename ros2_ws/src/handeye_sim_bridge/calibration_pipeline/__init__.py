"""Bilateral corner active hand-eye calibration.

The package is deliberately ROS-independent.  ROS nodes in
``handeye_sim_bridge`` adapt these components to TF, topics and robot motion.
"""

from .models import (
    BoardModel,
    CalibrationEstimate,
    CalibrationResult,
    Candidate,
    FlangePose,
    Measurement,
    Prediction,
    SensorROI,
)
from .pipeline import ActiveCalibrationPipeline, PipelineStage

__all__ = [
    "ActiveCalibrationPipeline",
    "BoardModel",
    "CalibrationEstimate",
    "CalibrationResult",
    "Candidate",
    "FlangePose",
    "Measurement",
    "PipelineStage",
    "Prediction",
    "SensorROI",
]
