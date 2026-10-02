#!/usr/bin/env python3
"""Generate a disposable simulation URDF with the YAML hand-eye truth.

The checked-in URDF remains a nominal template; this script never edits it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from scipy.spatial.transform import Rotation


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "ros2_ws/src/handeye_calibration_core"))
from calibration_pipeline.simulation.scene_truth import load_handeye_truth  # noqa: E402


def make_robot_urdf(config: Path, template: Path) -> ET.ElementTree:
    rotation, translation = load_handeye_truth(config)
    tree = ET.parse(template)
    origin = tree.find(
        "./joint[@name='fanuc_flange-gocator_sensor_joint']/origin"
    )
    if origin is None:
        raise ValueError("URDF lacks the fanuc_flange-gocator_sensor fixed joint")
    rpy = Rotation.from_matrix(rotation).as_euler("xyz")
    origin.set("xyz", " ".join(f"{value:.12g}" for value in translation))
    origin.set("rpy", " ".join(f"{value:.12g}" for value in rpy))
    return tree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=WORKSPACE / "ros2_ws/src/handeye_sim_bridge/config/calibration.yaml",
    )
    parser.add_argument(
        "--template", type=Path, default=WORKSPACE / "urdf/calib_robot.urdf"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output in {args.config.resolve(), args.template.resolve()}:
        parser.error("output must not overwrite the configuration or source URDF")
    tree = make_robot_urdf(args.config, args.template)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="unicode", xml_declaration=True)
    print(f"Generated YAML-truth simulation URDF: {output}")


if __name__ == "__main__":
    main()
