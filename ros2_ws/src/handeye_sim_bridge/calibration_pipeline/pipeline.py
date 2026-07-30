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
from .validation import ValidationMetrics, held_out_geometric_metrics


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
        initial_maximum_update_rotation_deg: float | None = None,
        initial_maximum_update_translation_m: float | None = None,
        initial_maximum_board_rotation_deg: float | None = None,
        validation_minimum_relative_improvement: float = 0.01,
        validation_patience: int = 3,
        validation_best_relative_tolerance: float = 0.10,
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
        # The six-seed solution can be deliberately coarse under a demanding
        # disturbance test.  Its first independent NBV is therefore allowed
        # to make a larger correction, bounded by the configured Phase-0a
        # uncertainty.  Once one NBV has been committed, the tighter
        # transactional limits apply again.
        self.initial_maximum_update_rotation_deg = float(
            maximum_update_rotation_deg
            if initial_maximum_update_rotation_deg is None
            else initial_maximum_update_rotation_deg
        )
        self.initial_maximum_update_translation_m = float(
            maximum_update_translation_m
            if initial_maximum_update_translation_m is None
            else initial_maximum_update_translation_m
        )
        self.initial_maximum_board_rotation_deg = float(
            maximum_board_rotation_deg
            if initial_maximum_board_rotation_deg is None
            else initial_maximum_board_rotation_deg
        )
        for name, value in (
            ("maximum_update_rotation_deg", self.maximum_update_rotation_deg),
            ("maximum_update_translation_m", self.maximum_update_translation_m),
            ("maximum_board_rotation_deg", self.maximum_board_rotation_deg),
            (
                "initial_maximum_update_rotation_deg",
                self.initial_maximum_update_rotation_deg,
            ),
            (
                "initial_maximum_update_translation_m",
                self.initial_maximum_update_translation_m,
            ),
            (
                "initial_maximum_board_rotation_deg",
                self.initial_maximum_board_rotation_deg,
            ),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.stage = PipelineStage.WAIT_MANUAL_INIT
        self.poses: list[FlangePose] = []
        self.measurements: list[Measurement] = []
        self.seed_count = 0
        self.nbv_count = 0
        self.result: CalibrationResult | None = None
        self.failed_candidates: set[str] = set()
        self.gain_history: list[float] = []
        self.validation_poses: list[FlangePose] = []
        self.validation_measurements: list[Measurement] = []
        self.result_history: list[tuple[int, CalibrationResult]] = []
        self.validation_metrics_history: list[ValidationMetrics | None] = []
        self.best_result: CalibrationResult | None = None
        self.best_result_nbv_index = 0
        self.current_validation_metrics: ValidationMetrics | None = None
        self.best_validation_metrics: ValidationMetrics | None = None
        self.validation_minimum_relative_improvement = float(
            validation_minimum_relative_improvement
        )
        self.validation_patience = int(validation_patience)
        self.validation_best_relative_tolerance = float(
            validation_best_relative_tolerance
        )
        self.validation_no_improvement_count = 0
        if not 0.0 <= self.validation_minimum_relative_improvement < 1.0:
            raise ValueError(
                "validation_minimum_relative_improvement must be in [0, 1)"
            )
        if self.validation_patience < 1:
            raise ValueError("validation_patience must be positive")
        if not 0.0 <= self.validation_best_relative_tolerance < 1.0:
            raise ValueError(
                "validation_best_relative_tolerance must be in [0, 1)"
            )

    def append_seed(self, pose: FlangePose, measurement: Measurement) -> None:
        self.append_seed_batch([pose], [measurement])

    def append_seed_batch(
        self,
        poses: list[FlangePose],
        measurements: list[Measurement],
    ) -> None:
        """Append all synchronized frames from one physical seed pose."""
        if self.stage not in {PipelineStage.WAIT_MANUAL_INIT, PipelineStage.COLLECT_SEEDS}:
            raise RuntimeError(f"cannot append a seed during {self.stage.name}")
        if not poses or len(poses) != len(measurements):
            raise ValueError(
                "seed batch must contain equal non-empty pose/measurement lists"
            )
        self.poses.extend(poses)
        self.measurements.extend(measurements)
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
        self._register_result(self.result)
        return self.result

    def append_validation_observation(
        self, pose: FlangePose, measurement: Measurement
    ) -> None:
        """Add one held-out physical-pose representative.

        Validation observations never enter the solve or the information
        matrix.  Seed validation can therefore be populated before
        :meth:`initialize`.
        """
        self.validation_poses.append(pose)
        self.validation_measurements.append(measurement)

    def _register_result(self, result: CalibrationResult) -> None:
        self.result_history.append((self.nbv_count, result))
        metrics = [
            held_out_geometric_metrics(
                item,
                self.validation_poses,
                self.validation_measurements,
                weights=getattr(self.solver, "weights", None),
            )
            for _index, item in self.result_history
        ]
        self.validation_metrics_history = metrics
        current = metrics[-1]
        self.current_validation_metrics = current
        finite = [
            (index, metric)
            for index, metric in enumerate(metrics)
            if metric is not None and np.isfinite(metric.score_m)
        ]
        if not finite:
            self.best_result = result
            self.best_result_nbv_index = self.nbv_count
            self.best_validation_metrics = None
            return
        minimum_score = min(metric.score_m for _index, metric in finite)
        statistically_equivalent = [
            (index, metric)
            for index, metric in finite
            if metric.score_m
            <= minimum_score * (1.0 + self.validation_best_relative_tolerance)
        ]
        # Held-out scores based on only a few noisy physical poses fluctuate.
        # Prefer the most informed/newest snapshot inside a configurable
        # equivalence band; roll back only for a material regression.
        best_history_index, best_metric = max(
            statistically_equivalent, key=lambda item: item[0]
        )
        self.best_result_nbv_index, self.best_result = self.result_history[
            best_history_index
        ]
        self.best_validation_metrics = best_metric
        if len(metrics) <= 1 or current is None:
            self.validation_no_improvement_count = 0
            return
        previous_scores = [
            metric.score_m
            for metric in metrics[:-1]
            if metric is not None and np.isfinite(metric.score_m)
        ]
        if not previous_scores:
            self.validation_no_improvement_count = 0
            return
        previous_best = min(previous_scores)
        required = previous_best * (
            1.0 - self.validation_minimum_relative_improvement
        )
        if current.score_m < required:
            self.validation_no_improvement_count = 0
        else:
            self.validation_no_improvement_count += 1

    @property
    def validation_plateaued(self) -> bool:
        return (
            self.current_validation_metrics is not None
            and self.validation_no_improvement_count >= self.validation_patience
        )

    def restore_historical_best(self) -> CalibrationResult:
        """Select the newest snapshot statistically tied for held-out best."""
        if self.best_result is None:
            if self.result is None:
                raise RuntimeError("no calibration result is available")
            self.best_result = self.result
            self.best_result_nbv_index = self.nbv_count
        self.result = self.best_result
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
        minimum_valid_probability: float = 0.8,
        virtual_batch_size: int = 1,
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
            minimum_valid_probability=minimum_valid_probability,
            virtual_batch_size=virtual_batch_size,
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
        validation_pose: FlangePose | None = None,
        validation_measurement: Measurement | None = None,
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
        if (validation_pose is None) != (validation_measurement is None):
            raise ValueError(
                "validation_pose and validation_measurement must be provided together"
            )
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
        first_nbv = self.nbv_count == 0
        rotation_limit = (
            self.initial_maximum_update_rotation_deg
            if first_nbv
            else self.maximum_update_rotation_deg
        )
        translation_limit = (
            self.initial_maximum_update_translation_m
            if first_nbv
            else self.maximum_update_translation_m
        )
        board_limit = (
            self.initial_maximum_board_rotation_deg
            if first_nbv
            else self.maximum_board_rotation_deg
        )
        if (
            rotation_jump > rotation_limit
            or translation_jump > translation_limit
            or board_jump > board_limit
        ):
            raise RuntimeError(
                "NBV observation rejected: transactional update jump "
                f"(handeye={rotation_jump:.3f} deg/{translation_jump:.4f} m, "
                f"board={board_jump:.3f} deg; "
                f"limits={rotation_limit:.3f} deg/{translation_limit:.4f} m/"
                f"{board_limit:.3f} deg; "
                f"phase={'first-NBV correction' if first_nbv else 'rolling update'})"
            )
        self.poses = trial_poses
        self.measurements = trial_measurements
        self.nbv_count += 1
        self.result = trial
        if validation_pose is not None and validation_measurement is not None:
            self.append_validation_observation(
                validation_pose, validation_measurement
            )
        self._register_result(trial)
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
            validation_plateaued=self.validation_plateaued,
        )
        if stop:
            self.stage = PipelineStage.COMPLETE
        return stop, reason
