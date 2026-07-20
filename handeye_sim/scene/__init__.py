#!/usr/bin/env python3
"""scene/__init__.py"""
from handeye_sim.scene.fov_geometry import (
    generate_hand_eye_gt, generate_plane,
    compute_fov_plate_scanline, compute_fov_triangle,
    build_R_edge, collect_frames, make_transform,
)
