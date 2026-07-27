"""High-level state and data ownership for the complete active pipeline."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

from .geometry import rotation_distance_deg
from .models import CalibrationResult, Candidate, FlangePose, Measurement, SensorROI
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
        maximum_update_rotation_deg: float = 5.0,
        maximum_update_translation_m: float = 0.05,
        maximum_board_rotation_deg: float = 10.0,
    ) -> None:
        self.coarse_handeye_rotation_init = np.asarray(
            nominal_handeye_rotation, dtype=float
        )
        self.coarse_handeye_translation_init = np.asarray(
            nominal_handeye_translation, dtype=float
        )
        # Backward-compatible public names; both mean Phase-0a coarse input.
        self.nominal_handeye_rotation = self.coarse_handeye_rotation_init
        self.nominal_handeye_translation = self.coarse_handeye_translation_init
        self.board_dimensions = board_dimensions
        self.roi = roi or SensorROI()
        self.solver = solver or TwelveDofV2Solver()
        self.stop_policy = stop_policy or StopPolicy()
        self.minimum_seed_poses = minimum_seed_poses
        self.maximum_update_rotation_deg = float(maximum_update_rotation_deg)
        self.maximum_update_translation_m = float(maximum_update_translation_m)
        self.maximum_board_rotation_deg = float(maximum_board_rotation_deg)
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

    @staticmethod
    def _uniform_candidate_subset(
        candidates: list[Candidate], maximum_candidates: int | None
    ) -> list[Candidate]:
        """Keep coverage of the full deterministic grid instead of its prefix."""
        if maximum_candidates is None or len(candidates) <= maximum_candidates:
            return candidates
        indices = np.linspace(
            0, len(candidates) - 1, maximum_candidates, dtype=int
        )
        return [candidates[int(index)] for index in np.unique(indices)]

    def rank_candidates(
        self,
        *,
        maximum_candidates: int | None = None,
        candidate_filter=None,
        candidate_options: dict | None = None,
    ):
        if self.stage is not PipelineStage.ACTIVE_NBV or self.result is None:
            raise RuntimeError("initialize a valid 12-DOF-V2 estimate before NBV")
        candidates = [
            candidate
            for candidate in generate_candidates(
                self.result.estimate,
                roi=self.roi,
                **(candidate_options or {}),
            )
            if candidate.candidate_id not in self.failed_candidates
        ]
        if candidate_filter is not None:
            candidates = [
                candidate for candidate in candidates if candidate_filter(candidate)
            ]
        candidates = self._uniform_candidate_subset(
            candidates, maximum_candidates
        )
        return score_candidates(
            candidates,
            self.result,
            self.poses,
            self.measurements,
            self.roi,
            maximum_candidates=None,
            projection_weights=self.solver.weights,
            state_scale=self.solver.state_scale,
        )

    def reject_candidate(self, candidate_id: str) -> None:
        self.failed_candidates.add(candidate_id)

    def exclude_candidate(self, candidate_id: str) -> None:
        """Do not score an already observed deterministic grid candidate again."""
        self.failed_candidates.add(candidate_id)

    def append_nbv(
        self,
        pose: FlangePose,
        measurement: Measurement,
        *,
        candidate_id: str | None = None,
    ) -> CalibrationResult:
        return self.append_nbv_batch(
            [pose], [measurement], candidate_id=candidate_id
        )

    def append_nbv_batch(
        self,
        poses: list[FlangePose],
        measurements: list[Measurement],
        *,
        candidate_id: str | None = None,
    ) -> CalibrationResult:
        """Commit one physical NBV containing trigger-synchronized frames.

        Every frame retains its own encoder pose.  Treating a multi-frame
        batch as one concatenated profile with only the final pose creates a
        model error whenever the controller is still settling by a small
        amount.
        """
        if self.stage is not PipelineStage.ACTIVE_NBV:
            raise RuntimeError(f"cannot append an NBV observation during {self.stage.name}")
        if not poses or len(poses) != len(measurements):
            raise ValueError("NBV batch must contain equal non-empty pose/measurement lists")
        for measurement in measurements:
            if not self.roi.contains(
                measurement.endpoint_u, safe=False
            ) or not self.roi.contains(
                measurement.endpoint_v, safe=False
            ):
                raise RuntimeError(
                    "NBV observation rejected: endpoints outside hard valid domain"
                )
        previous = self.result
        trial_poses = self.poses + list(poses)
        trial_measurements = self.measurements + list(measurements)
        trial = self.solver.solve(
            trial_poses,
            trial_measurements,
            self.result.estimate.handeye_rotation,
            self.result.estimate.handeye_translation,
            board_dimensions=self.board_dimensions,
            initial_board_rotation=self.result.estimate.board.rotation,
        )
        if not trial.converged:
            raise RuntimeError("NBV observation rejected: trial solve is not observable")
        rotation_jump = rotation_distance_deg(
            previous.estimate.handeye_rotation, trial.estimate.handeye_rotation
        )
        translation_jump = float(
            np.linalg.norm(
                previous.estimate.handeye_translation
                - trial.estimate.handeye_translation
            )
        )
        board_jump = rotation_distance_deg(
            previous.estimate.board.rotation, trial.estimate.board.rotation
        )
        if (
            rotation_jump > self.maximum_update_rotation_deg
            or translation_jump > self.maximum_update_translation_m
            or board_jump > self.maximum_board_rotation_deg
        ):
            raise RuntimeError(
                "NBV observation rejected: transactional update jump "
                f"(handeye={rotation_jump:.3f} deg/{translation_jump:.4f} m, "
                f"board={board_jump:.3f} deg)"
            )
        self.poses = trial_poses
        self.measurements = trial_measurements
        self.nbv_count += 1
        self.result = trial
        if candidate_id is not None:
            self.exclude_candidate(candidate_id)
        return trial

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
            # A synchronized measurement batch is one physical robot pose,
            # even though it contributes multiple statistical observations.
            total_poses=self.seed_count + self.nbv_count,
            nbv_poses=self.nbv_count,
            effective_rank=rank,
            best_information_gain=best_gain,
            minimum_effective_eigenvalue=float(eigenvalues[0]),
            handeye_covariance=(
                None
                if self.result.estimate.covariance_x9 is None
                else self.result.estimate.covariance_x9[:6, :6]
            ),
        )
        if stop:
            self.stage = PipelineStage.COMPLETE
        return stop, reason
