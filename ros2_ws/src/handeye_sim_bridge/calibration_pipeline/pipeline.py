"""High-level state and data ownership for the complete active pipeline."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

from .models import CalibrationResult, FlangePose, Measurement, SensorROI
from .nbv.candidate_generator import generate_candidates
from .nbv.scoring import score_candidates
from .nbv.stopping import StopPolicy
from .solvers import TwelveDofV2Solver


class PipelineStage(Enum):
    WAIT_MANUAL_INIT = auto()
    COLLECT_SEEDS = auto()
    INITIALIZE_12DOF_V2 = auto()
    ACTIVE_NBV = auto()
    COMPLETE = auto()
    FAILED = auto()


class ActiveCalibrationPipeline:
    """Own observations and coordinate solver/NBV decisions.

    Robot motion is intentionally outside this class.  A ROS or hardware
    adapter executes the returned fixed flange command and appends only a
    verified bilateral observation.
    """

    def __init__(
        self,
        nominal_handeye_rotation: np.ndarray,
        nominal_handeye_translation: np.ndarray,
        board_dimensions: tuple[float, float],
        *,
        roi: SensorROI | None = None,
        solver: TwelveDofV2Solver | None = None,
        stop_policy: StopPolicy | None = None,
        minimum_seed_poses: int = 6,
    ) -> None:
        self.nominal_handeye_rotation = np.asarray(nominal_handeye_rotation, dtype=float)
        self.nominal_handeye_translation = np.asarray(nominal_handeye_translation, dtype=float)
        self.board_dimensions = board_dimensions
        self.roi = roi or SensorROI()
        self.solver = solver or TwelveDofV2Solver()
        self.stop_policy = stop_policy or StopPolicy()
        self.minimum_seed_poses = minimum_seed_poses
        self.stage = PipelineStage.WAIT_MANUAL_INIT
        self.poses: list[FlangePose] = []
        self.measurements: list[Measurement] = []
        self.seed_count = 0
        self.nbv_count = 0
        self.result: CalibrationResult | None = None
        self.failed_candidates: set[str] = set()
        self.gain_history: list[float] = []

    def append_seed(self, pose: FlangePose, measurement: Measurement) -> None:
        if self.stage not in {PipelineStage.WAIT_MANUAL_INIT, PipelineStage.COLLECT_SEEDS}:
            raise RuntimeError(f"cannot append a seed during {self.stage.name}")
        self.poses.append(pose)
        self.measurements.append(measurement)
        self.seed_count += 1
        self.stage = (
            PipelineStage.INITIALIZE_12DOF_V2
            if self.seed_count >= self.minimum_seed_poses
            else PipelineStage.COLLECT_SEEDS
        )

    def initialize(self) -> CalibrationResult:
        if self.seed_count < self.minimum_seed_poses:
            raise RuntimeError(
                f"need {self.minimum_seed_poses} seeds, currently have {self.seed_count}"
            )
        self.result = self.solver.solve(
            self.poses,
            self.measurements,
            self.nominal_handeye_rotation,
            self.nominal_handeye_translation,
            board_dimensions=self.board_dimensions,
        )
        if not self.result.converged:
            self.stage = PipelineStage.COLLECT_SEEDS
            return self.result
        self.stage = PipelineStage.ACTIVE_NBV
        return self.result

    def rank_candidates(self, *, maximum_candidates: int | None = None):
        if self.stage is not PipelineStage.ACTIVE_NBV or self.result is None:
            raise RuntimeError("initialize a valid 12-DOF-V2 estimate before NBV")
        candidates = [
            candidate
            for candidate in generate_candidates(self.result.estimate)
            if candidate.candidate_id not in self.failed_candidates
        ]
        return score_candidates(
            candidates,
            self.result,
            self.poses,
            self.measurements,
            self.roi,
            maximum_candidates=maximum_candidates,
            projection_weights=self.solver.weights,
        )

    def reject_candidate(self, candidate_id: str) -> None:
        self.failed_candidates.add(candidate_id)

    def append_nbv(self, pose: FlangePose, measurement: Measurement) -> CalibrationResult:
        if self.stage is not PipelineStage.ACTIVE_NBV:
            raise RuntimeError(f"cannot append an NBV observation during {self.stage.name}")
        self.poses.append(pose)
        self.measurements.append(measurement)
        self.nbv_count += 1
        self.result = self.solver.solve(
            self.poses,
            self.measurements,
            self.result.estimate.handeye_rotation,
            self.result.estimate.handeye_translation,
            board_dimensions=self.board_dimensions,
            initial_board_rotation=self.result.estimate.board.rotation,
        )
        return self.result

    def check_stop(self, ranked_candidates) -> tuple[bool, str]:
        if self.result is None:
            return False, "not initialized"
        if ranked_candidates:
            best_gain = ranked_candidates[0].information_gain
        else:
            best_gain = float("-inf")
        self.gain_history.append(best_gain)
        effective = self.result.diagnostics.effective_handeye_information
        eigenvalues = np.linalg.eigvalsh(effective)
        rank = int(np.linalg.matrix_rank(effective))
        stop, reason = self.stop_policy.evaluate(
            total_poses=len(self.poses),
            nbv_poses=self.nbv_count,
            effective_rank=rank,
            best_information_gain=best_gain,
            minimum_effective_eigenvalue=float(eigenvalues[0]),
        )
        if stop:
            self.stage = PipelineStage.COMPLETE
        return stop, reason
