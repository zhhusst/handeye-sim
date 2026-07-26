#!/usr/bin/env python3
"""Container dependency and package smoke test."""

import matplotlib
import numpy as np
import rclpy
import scipy
import yaml

from calibration_pipeline import ActiveCalibrationPipeline
from calibration_pipeline.solvers import TwelveDofV2Solver

assert ActiveCalibrationPipeline
assert TwelveDofV2Solver
rclpy.init()
rclpy.shutdown()
print("all imports OK")
print(f"numpy={np.__version__}")
print(f"scipy={scipy.__version__}")
print(f"matplotlib={matplotlib.__version__}")
print(f"yaml={yaml.__version__}")
