#!/usr/bin/env python3
"""Deterministic batch robustness study for Phase-0b initial poses.

This is a simulation-only validation utility.  Ground-truth hand-eye geometry
is used solely to render virtual Gocator observations; the qualification and
servo logic receive only profiles, flange poses and local kinematics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


WORKSPACE = Path("/workspace")
CORE_PACKAGE_ROOT = WORKSPACE / "ros2_ws/src/handeye_calibration_core"
ROS_PACKAGE_ROOT = WORKSPACE / "ros2_ws/src/handeye_sim_bridge"
FANUC_SUPPORT_ROOT = WORKSPACE / "ros2_ws/src/fanuc_m20id25_support"
SIM_BACKEND_ROOT = WORKSPACE / "ros2_ws/src/handeye_sim_backend"
sys.path.insert(0, str(ROS_PACKAGE_ROOT))
sys.path.insert(0, str(FANUC_SUPPORT_ROOT))
sys.path.insert(0, str(CORE_PACKAGE_ROOT))

from calibration_pipeline.geometry import (
    invert_transform,
    make_transform,
    rotation_distance_deg,
    so3_exp,
)
from calibration_pipeline.models import SensorROI, TrapezoidDomain
from calibration_pipeline.seed_collection import (
    InitialPoseCriteria,
    TranslationServo,
    adaptive_rotation_plan,
    assess_initial_pose,
    evaluate_bilateral_feature,
    rotation_diversity,
    seed_feature_is_acceptable,
)
from calibration_pipeline.nbv.candidate_generator import _sensor_transform
from calibration_pipeline.simulation.scanline import compute_fov_plate_scanline
from calibration_pipeline.simulation.scene_truth import (
    HAND_EYE_ROTATION,
    HAND_EYE_TRANSLATION,
)
from fanuc_m20id25_support.fanuc_kinematic import (
    JOINT_LIMITS_DEG,
    forward_kinematics_urdf,
    inverse_kinematics_numeric,
)


REFERENCE_JOINTS = np.array(
    [-0.2357, -0.0364, -0.6328, -0.4062, -1.0504, 0.8788]
)
OBSERVED_CASES = {
    "known_good": REFERENCE_JOINTS,
    "failed_reversed_depth": np.array(
        [-0.8643, -0.2791, -1.1358, -1.1780, -1.3738, 2.7938]
    ),
    "failed_far_view": np.array(
        [0.377, -0.159, -0.307, 0.194, -1.241, 0.393]
    ),
}


def configured_roi() -> SensorROI:
    return SensorROI(
        hard_domain=TrapezoidDomain(
            -0.292, 0.82, -0.021, -0.22, -0.019, 0.22
        ),
        safe_domain=TrapezoidDomain(
            0.27, 0.78, -0.115, -0.19, 0.095, 0.19
        ),
    )


class VirtualSeedStudy:
    def __init__(self, *, rotation_target_deg: float = 6.0) -> None:
        self.roi = configured_roi()
        self.handeye = make_transform(
            HAND_EYE_ROTATION, HAND_EYE_TRANSLATION
        )
        factory = json.loads(
            (
                SIM_BACKEND_ROOT / "config/fov_factory_calib.json"
            ).read_text(encoding="utf-8")
        )
        self.fov_corners = np.asarray(factory["fov_corners_S"], dtype=float)
        self.joint_limits = np.deg2rad(JOINT_LIMITS_DEG)
        self.criteria = InitialPoseCriteria()
        self.rotation_target = np.deg2rad(rotation_target_deg)

    def observe(self, flange: np.ndarray):
        sensor = flange @ self.handeye
        result = compute_fov_plate_scanline(
            sensor[:3, :3],
            sensor[:3, 3],
            np.array([0.7, 0.0, 0.25]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            0.4,
            0.5,
            n_sample=240,
            fov_corners_S=self.fov_corners,
        )
        endpoints = dict(result["endpoints_S"])
        if "e1" not in endpoints or "e2" not in endpoints:
            return None
        return evaluate_bilateral_feature(
            endpoints["e1"], endpoints["e2"], self.roi
        )

    @staticmethod
    def local_ik_coverage(flange: np.ndarray, joints: np.ndarray) -> int:
        feasible = 0
        for axis in (0, 1):
            for sign in (-1, 1):
                vector = np.zeros(3)
                vector[axis] = sign * np.deg2rad(2.0)
                target = flange.copy()
                target[:3, :3] = flange[:3, :3] @ so3_exp(vector)
                solutions = inverse_kinematics_numeric(target, q_init=joints)
                if (
                    len(solutions) > 0
                    and np.max(np.abs(solutions[0] - joints))
                    <= np.deg2rad(20.0)
                ):
                    feasible += 1
        return feasible

    def assess(self, joints: np.ndarray):
        flange = forward_kinematics_urdf(joints)
        feature = self.observe(flange)
        if feature is None:
            return flange, None, None
        assessment = assess_initial_pose(
            feature,
            joints,
            self.joint_limits,
            local_ik_directions=self.local_ik_coverage(flange, joints),
            criteria=self.criteria,
        )
        return flange, feature, assessment

    def center(self, flange: np.ndarray):
        """Idealized measured local servo with safety backtracking."""
        current = flange.copy()
        for _ in range(8):
            feature = self.observe(current)
            if feature is None or not feature.safe:
                return None
            if abs(feature.x_mid) <= 0.003:
                return current, feature
            sensitivities: dict[int, float] = {}
            for axis in range(3):
                probe = current.copy()
                local = np.zeros(3)
                local[axis] = 0.001
                probe[:3, 3] += current[:3, :3] @ local
                probe_feature = self.observe(probe)
                if probe_feature is None or not probe_feature.safe:
                    continue
                sensitivities[axis] = (
                    probe_feature.x_mid - feature.x_mid
                ) / 0.001
            servo = TranslationServo(maximum_step=0.005)
            try:
                servo.choose_axis(sensitivities)
            except ValueError:
                return None
            step = servo.correction(feature.x_mid)
            accepted = None
            for scale in (1.0, 0.5, 0.25):
                proposal = current.copy()
                local = np.zeros(3)
                local[servo.axis] = scale * step
                proposal[:3, 3] += current[:3, :3] @ local
                proposal_feature = self.observe(proposal)
                if proposal_feature is not None and proposal_feature.safe:
                    accepted = proposal
                    break
            if accepted is None:
                return None
            current = accepted
        feature = self.observe(current)
        if (
            feature is not None
            and abs(feature.x_mid) <= 0.003
            and feature.safe
        ):
            return current, feature
        return None

    def preflight(self, reference: np.ndarray) -> dict:
        results = []
        for axis, axis_name in ((0, "x"), (1, "y")):
            for sign in (-1, 1):
                vector = np.zeros(3)
                vector[axis] = sign * np.deg2rad(2.0)
                proposal = reference.copy()
                proposal[:3, :3] = (
                    reference[:3, :3] @ so3_exp(vector)
                )
                centered = self.center(proposal)
                if centered is None:
                    accepted = False
                    margin = None
                else:
                    _, feature = centered
                    margin = float(feature.domain_margin)
                    accepted = seed_feature_is_acceptable(
                        feature,
                        maximum_abs_x_mid_m=0.003,
                        minimum_domain_margin_m=0.015,
                    )
                results.append(
                    {
                        "axis": axis,
                        "name": f"{axis_name}_{'positive' if sign > 0 else 'negative'}",
                        "accepted": accepted,
                        "domain_margin_m": margin,
                    }
                )
        feasible = [item for item in results if item["accepted"]]
        axes = {item["axis"] for item in feasible}
        return {
            "accepted": len(feasible) >= 3 and axes == {0, 1},
            "feasible_directions": len(feasible),
            "results": results,
        }

    def collect(self, reference: np.ndarray) -> dict:
        reference_feature = self.observe(reference)
        if reference_feature is None or not reference_feature.safe:
            return {"success": False, "seed_count": 0, "labels": []}
        rotations = [reference[:3, :3].copy()]
        labels = ["reference"]
        for target in adaptive_rotation_plan():
            if len(labels) >= 6:
                break
            current = reference.copy()
            last_valid = reference.copy()
            last_feature = reference_feature
            failures = 0
            step = np.deg2rad(2.0)
            target_complete = True
            for axis, sign in target.stages:
                accumulated = 0.0
                while accumulated + 1e-10 < self.rotation_target:
                    magnitude = min(
                        step, self.rotation_target - accumulated
                    )
                    vector = np.zeros(3)
                    vector[axis] = sign * magnitude
                    proposal = current.copy()
                    proposal[:3, :3] = (
                        current[:3, :3] @ so3_exp(vector)
                    )
                    centered = self.center(proposal)
                    if centered is None:
                        failures += 1
                        step = max(step / 2.0, np.deg2rad(0.25))
                        if failures >= 3:
                            target_complete = False
                            break
                        current = last_valid.copy()
                        continue
                    current, last_feature = centered
                    last_valid = current.copy()
                    accumulated += magnitude
                if not target_complete:
                    break
            candidate = current if target_complete else last_valid
            candidate_feature = (
                self.observe(candidate) if target_complete else last_feature
            )
            relative_angle = rotation_distance_deg(
                reference[:3, :3], candidate[:3, :3]
            )
            if (
                candidate_feature is None
                or relative_angle < 2.5
                or not seed_feature_is_acceptable(
                    candidate_feature,
                    maximum_abs_x_mid_m=0.003,
                    minimum_domain_margin_m=0.002,
                )
            ):
                continue
            trial = rotations + [candidate[:3, :3].copy()]
            if rotation_diversity(trial)["minimum_pairwise_deg"] < 2.0:
                continue
            rotations = trial
            labels.append(
                target.name if target_complete else f"{target.name}_partial"
            )
        diversity = rotation_diversity(rotations)
        return {
            "success": len(labels) >= 6,
            "seed_count": len(labels),
            "labels": labels,
            "minimum_pairwise_deg": float(
                diversity["minimum_pairwise_deg"]
            ),
            "minimum_gram_eigenvalue": float(
                diversity["minimum_gram_eigenvalue"]
            ),
        }


def sample_joints(
    study: VirtualSeedStudy, count: int, seed: int
) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    samples = [(name, joints.copy()) for name, joints in OBSERVED_CASES.items()]
    attempts = 0
    while len(samples) < count + len(OBSERVED_CASES) and attempts < count * 200:
        attempts += 1
        point_u = np.array([0.7 + rng.uniform(0.04, 0.22), 0.0, 0.25])
        point_v = np.array([0.7, rng.uniform(0.04, 0.24), 0.25])
        sensor = _sensor_transform(
            point_u,
            point_v,
            np.array([0.0, 0.0, 1.0]),
            np.deg2rad(rng.uniform(50.0, 82.0)),
            np.deg2rad(rng.uniform(-20.0, 20.0)),
            rng.uniform(0.30, 0.55),
            int(rng.choice((-1, 1))),
        )
        if sensor is None:
            continue
        rotation_sensor_base = sensor[:3, :3].T
        endpoint_u = rotation_sensor_base @ (point_u - sensor[:3, 3])
        endpoint_v = rotation_sensor_base @ (point_v - sensor[:3, 3])
        nominal_feature = evaluate_bilateral_feature(
            endpoint_u, endpoint_v, study.roi
        )
        # Generate cases inside the measurable portion of the proposed
        # envelope before spending time on numerical IK.
        if (
            not nominal_feature.safe
            or abs(nominal_feature.x_mid)
            > study.criteria.maximum_abs_x_mid_m
            or not (
                study.criteria.minimum_z_mid_m
                <= nominal_feature.z_mid
                <= study.criteria.maximum_z_mid_m
            )
            or nominal_feature.domain_margin
            < study.criteria.minimum_domain_margin_m
            or not (
                study.criteria.minimum_profile_length_m
                <= nominal_feature.profile_length
                <= study.criteria.maximum_profile_length_m
            )
            or abs(endpoint_v[2] - endpoint_u[2])
            < study.criteria.minimum_absolute_endpoint_depth_delta_m
        ):
            continue
        flange = sensor @ invert_transform(study.handeye)
        solutions = inverse_kinematics_numeric(
            flange, q_init=REFERENCE_JOINTS, max_iter=80
        )
        if len(solutions) == 0:
            continue
        joints = solutions[0]
        feature = study.observe(forward_kinematics_urdf(joints))
        if feature is None or not feature.safe:
            continue
        index = len(samples) - len(OBSERVED_CASES)
        samples.append((f"geometry_{index:03d}", joints))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rotation-target-deg", type=float, default=6.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "data/initial_pose_robustness.json",
    )
    args = parser.parse_args()
    study = VirtualSeedStudy(rotation_target_deg=args.rotation_target_deg)
    records = []
    cases = sample_joints(study, args.samples, args.seed)
    for name, joints in cases:
        flange, feature, assessment = study.assess(joints)
        collection = (
            study.collect(flange)
            if feature is not None and feature.safe
            else {"success": False, "seed_count": 0, "labels": []}
        )
        preflight = (
            study.preflight(flange)
            if feature is not None and feature.safe
            else {
                "accepted": False,
                "feasible_directions": 0,
                "results": [],
            }
        )
        record = {
            "name": name,
            "joints_rad": joints.tolist(),
            "bilateral_safe": bool(feature is not None and feature.safe),
            "envelope_accepted": bool(
                assessment is not None and assessment.accepted
            ),
            "reasons": (
                ["measurement_missing"]
                if assessment is None
                else list(assessment.reasons)
            ),
            "x_mid_mm": (
                None if feature is None else 1000.0 * feature.x_mid
            ),
            "z_mid_mm": (
                None if feature is None else 1000.0 * feature.z_mid
            ),
            "domain_margin_mm": (
                None if feature is None else 1000.0 * feature.domain_margin
            ),
            "endpoint_depth_delta_mm": (
                None
                if feature is None
                else 1000.0
                * (feature.endpoint_v[2] - feature.endpoint_u[2])
            ),
            "absolute_endpoint_depth_delta_mm": (
                None
                if feature is None
                else 1000.0
                * abs(feature.endpoint_v[2] - feature.endpoint_u[2])
            ),
            "collection": collection,
            "preflight": preflight,
        }
        record["qualified"] = bool(
            record["envelope_accepted"] and preflight["accepted"]
        )
        records.append(record)
        print(
            f"{name:24s} safe={record['bilateral_safe']} "
            f"envelope={record['envelope_accepted']} "
            f"preflight={preflight['feasible_directions']}/4 "
            f"seeds={collection['seed_count']} "
            f"success={collection['success']}"
        )
    accepted = [item for item in records if item["envelope_accepted"]]
    qualified = [item for item in records if item["qualified"]]
    summary = {
        "requested_geometry_samples": args.samples,
        "generated_geometry_samples": len(cases) - len(OBSERVED_CASES),
        "total_cases": len(records),
        "bilateral_safe_cases": sum(
            item["bilateral_safe"] for item in records
        ),
        "envelope_accepted_cases": len(accepted),
        "accepted_collection_successes": sum(
            item["collection"]["success"] for item in accepted
        ),
        "accepted_collection_failures": sum(
            not item["collection"]["success"] for item in accepted
        ),
        "accepted_success_rate": (
            0.0
            if not accepted
            else sum(item["collection"]["success"] for item in accepted)
            / len(accepted)
        ),
        "fully_qualified_cases": len(qualified),
        "qualified_collection_successes": sum(
            item["collection"]["success"] for item in qualified
        ),
        "qualified_collection_failures": sum(
            not item["collection"]["success"] for item in qualified
        ),
        "qualified_success_rate": (
            0.0
            if not qualified
            else sum(item["collection"]["success"] for item in qualified)
            / len(qualified)
        ),
    }
    payload = {
        "schema_version": 1,
        "simulation_only": True,
        "criteria": study.criteria.__dict__,
        "rotation_target_deg": args.rotation_target_deg,
        "summary": summary,
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
