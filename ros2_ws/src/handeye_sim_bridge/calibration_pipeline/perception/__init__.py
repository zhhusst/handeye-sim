"""Sensor-profile perception shared by simulation and real ROS adapters."""

from .endpoint_detector import (
    EndpointDetection,
    EndpointDetectionConfig,
    ProfileEndpointDetector,
)

__all__ = [
    "EndpointDetection",
    "EndpointDetectionConfig",
    "ProfileEndpointDetector",
]
