"""Unified flat/shared-shape 12-DOF-V2 calibration backend.

The geometric core remains the 12 physical DOF (six hand-eye plus six target
frame DOF).  In ``shared`` mode a low-dimensional, view-invariant height field
is estimated as nuisance state and marginalized when hand-eye information is
computed.  ``flat`` is retained as an ablation and compatibility mode.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from ..geometry import so3_exp, so3_log
from ..models import (
    BoardModel,
    CalibrationEstimate,
    CalibrationResult,
    FlangePose,
    Measurement,
    SolverDiagnostics,
)
from ..v2_backend.corner_projection import solve_corner
from ..v2_backend.information import (
    DEFAULT_STATE_SCALE,
    covariance_from_jacobian,
    effective_handeye_information,
    observability,
    scaled_jacobian,
    validate_state_scale,
)
from ..v2_backend.plane_frame import canonicalize_plane_frame
from ..v2_backend.residual import numerical_jacobian, variable_projection_residual
from ..v2_backend.shared_surface import (
    SurfaceBasis,
    get_surface_basis,
    shared_surface_residual,
)


def _fit_direction(points: list[np.ndarray]) -> np.ndarray:
    if len(points) < 2:
        raise ValueError("at least two endpoints are required for each physical edge")
    values = np.asarray(points)
    _, _, right = np.linalg.svd(values - values.mean(axis=0), full_matrices=False)
    return right[0] / np.linalg.norm(right[0])


def _initial_board_rotation(
    poses: list[FlangePose],
    measurements: list[Measurement],
    handeye_rotation: np.ndarray,
    handeye_translation: np.ndarray,
) -> np.ndarray:
    endpoints_u: list[np.ndarray] = []
    endpoints_v: list[np.ndarray] = []
    for pose, measurement in zip(poses, measurements):
        sensor_rotation = pose.rotation @ handeye_rotation
        sensor_translation = pose.translation + pose.rotation @ handeye_translation
        endpoints_u.append(sensor_rotation @ measurement.endpoint_u + sensor_translation)
        endpoints_v.append(sensor_rotation @ measurement.endpoint_v + sensor_translation)
    u = _fit_direction(endpoints_u)
    v_raw = _fit_direction(endpoints_v)
    v = v_raw - u * (u @ v_raw)
    if np.linalg.norm(v) < 1e-8:
        raise ValueError("initial endpoint directions are degenerate")
    v /= np.linalg.norm(v)
    normal = np.cross(u, v)
    normal /= np.linalg.norm(normal)
    v = np.cross(normal, u)
    return np.column_stack((u, v, normal))


class TwelveDofV2Solver:
    """Solve the original flat model or its shared-target-form extension."""

    def __init__(
        self,
        *,
        plane_weight: float = 1.0,
        edge_weight: float = 1.0,
        endpoint_plane_weight: float = 1.0,
        max_evaluations: int = 3000,
        tolerance: float = 1e-11,
        state_scale: np.ndarray | None = None,
        maximum_condition_number: float = 1e12,
        surface_model: str = "flat",
        surface_basis_kind: str = "legendre",
        surface_degree: int = 4,
        shape_scale_m: float = 5e-4,
        shape_regularization: float = 1e-2,
        multistart_enabled: bool = False,
        multistart_maximum_board_tilt_deg: float = 20.0,
        multistart_require_plausible: bool = True,
    ) -> None:
        if surface_model not in {"flat", "shared"}:
            raise ValueError("surface_model must be 'flat' or 'shared'")
        if shape_scale_m <= 0.0:
            raise ValueError("shape_scale_m must be positive")
        if shape_regularization < 0.0:
            raise ValueError("shape_regularization must be non-negative")
        if not 0.0 < multistart_maximum_board_tilt_deg <= 90.0:
            raise ValueError(
                "multistart_maximum_board_tilt_deg must lie in (0, 90]"
            )
        self.weights = {
            "plane_weight": float(plane_weight),
            "edge_weight": float(edge_weight),
            "endpoint_plane_weight": float(endpoint_plane_weight),
        }
        self.max_evaluations = int(max_evaluations)
        self.tolerance = float(tolerance)
        self.maximum_condition_number = float(maximum_condition_number)
        self.surface_model = str(surface_model)
        self.surface_basis_kind = str(surface_basis_kind)
        self.surface_degree = int(surface_degree)
        self.shape_scale_m = float(shape_scale_m)
        self.shape_regularization = float(shape_regularization)
        self.multistart_enabled = bool(multistart_enabled)
        self.multistart_maximum_board_tilt_deg = float(
            multistart_maximum_board_tilt_deg
        )
        self.multistart_require_plausible = bool(
            multistart_require_plausible
        )
        self.surface_basis: SurfaceBasis | None = (
            None
            if self.surface_model == "flat"
            else get_surface_basis(self.surface_basis_kind, self.surface_degree)
        )
        self.flat_state_scale = validate_state_scale(
            DEFAULT_STATE_SCALE if state_scale is None else state_scale
        )
        if self.surface_basis is None:
            self.state_scale = self.flat_state_scale
        else:
            self.state_scale = np.concatenate(
                (
                    self.flat_state_scale,
                    np.full(3, self.flat_state_scale[3]),
                    np.full(self.surface_basis.size, self.shape_scale_m),
                )
            )

    @property
    def uses_shared_surface(self) -> bool:
        return self.surface_model == "shared"

    def residual(
        self,
        state: np.ndarray,
        poses: list[FlangePose],
        measurements: list[Measurement],
        *,
        board_dimensions: tuple[float, float],
        include_regularization: bool = True,
    ) -> np.ndarray:
        """Evaluate exactly the model used by solve and NBV scoring."""
        if self.surface_basis is None:
            return variable_projection_residual(
                state, poses, measurements, **self.weights
            )
        return shared_surface_residual(
            state,
            poses,
            measurements,
            board_dimensions=board_dimensions,
            basis=self.surface_basis,
            plane_weight=self.weights["plane_weight"],
            edge_weight=self.weights["edge_weight"],
            endpoint_surface_weight=self.weights["endpoint_plane_weight"],
            shape_regularization=self.shape_regularization,
            include_regularization=include_regularization,
        )

    def observation_residual(
        self,
        state: np.ndarray,
        poses: list[FlangePose],
        measurements: list[Measurement],
        *,
        board_dimensions: tuple[float, float],
    ) -> np.ndarray:
        return self.residual(
            state,
            poses,
            measurements,
            board_dimensions=board_dimensions,
            include_regularization=False,
        )

    def solve(
        self,
        poses: list[FlangePose],
        measurements: list[Measurement],
        nominal_handeye_rotation: np.ndarray,
        nominal_handeye_translation: np.ndarray,
        *,
        board_dimensions: tuple[float, float],
        initial_board_rotation: np.ndarray | None = None,
        initial_estimate: CalibrationEstimate | None = None,
    ) -> CalibrationResult:
        # Only the first six-seed solve needs global basin discovery.  Rolling
        # NBV updates carry the last accepted estimate and deliberately remain
        # local so that a later observation cannot jump between discrete
        # hand-eye/board hypotheses.
        if self.multistart_enabled and initial_estimate is None:
            flat, attempts, selected_name, selected_tilt = (
                self._solve_flat_multistart(
                    poses,
                    measurements,
                    nominal_handeye_rotation,
                    nominal_handeye_translation,
                    board_dimensions=board_dimensions,
                    initial_board_rotation=initial_board_rotation,
                )
            )
            if self.surface_basis is None:
                return self._attach_initialization_diagnostics(
                    flat,
                    attempts,
                    selected_name,
                    selected_tilt,
                    flat.cost,
                )
            result = self._solve_shared(
                poses,
                measurements,
                nominal_handeye_rotation,
                nominal_handeye_translation,
                board_dimensions=board_dimensions,
                initial_board_rotation=initial_board_rotation,
                initial_estimate=flat.estimate,
            )
            final_tilt = self._board_tilt_deg(
                result.estimate.board.rotation
            )
            if final_tilt > self.multistart_maximum_board_tilt_deg:
                result = replace(
                    result,
                    converged=False,
                    message=(
                        f"{result.message}; shared refinement violated the "
                        "flat-board geometry gate: "
                        f"tilt={final_tilt:.3f} deg > "
                        f"{self.multistart_maximum_board_tilt_deg:.3f} deg"
                    ),
                )
            return self._attach_initialization_diagnostics(
                result,
                attempts,
                selected_name,
                selected_tilt,
                flat.cost,
            )
        if self.surface_basis is None:
            return self._solve_flat(
                poses,
                measurements,
                nominal_handeye_rotation,
                nominal_handeye_translation,
                board_dimensions=board_dimensions,
                initial_board_rotation=initial_board_rotation,
            )
        return self._solve_shared(
            poses,
            measurements,
            nominal_handeye_rotation,
            nominal_handeye_translation,
            board_dimensions=board_dimensions,
            initial_board_rotation=initial_board_rotation,
            initial_estimate=initial_estimate,
        )

    @staticmethod
    def _board_tilt_deg(board_rotation: np.ndarray) -> float:
        """Return plane-normal tilt from the base vertical axis.

        A plate may be freely yawed on the workbench.  Therefore the gate uses
        only ``|n_B dot z_base|`` and accepts either normal sign; constraining
        the full SO(3) distance would incorrectly require the plate edges to
        be aligned with the base X/Y axes.
        """
        normal = np.asarray(board_rotation, dtype=float)[:, 2]
        cosine = float(np.clip(abs(normal[2]), 0.0, 1.0))
        return float(np.rad2deg(np.arccos(cosine)))

    @staticmethod
    def _flat_rotation_hypotheses() -> tuple[tuple[str, np.ndarray], ...]:
        """Discrete proper rotations covering line-scanner axis flips."""
        return (
            ("nominal", np.eye(3)),
            ("local_x_180", np.diag([1.0, -1.0, -1.0])),
            ("local_y_180", np.diag([-1.0, 1.0, -1.0])),
            ("local_z_180", np.diag([-1.0, -1.0, 1.0])),
        )

    def _solve_flat_multistart(
        self,
        poses: list[FlangePose],
        measurements: list[Measurement],
        nominal_handeye_rotation: np.ndarray,
        nominal_handeye_translation: np.ndarray,
        *,
        board_dimensions: tuple[float, float],
        initial_board_rotation: np.ndarray | None,
    ) -> tuple[
        CalibrationResult,
        tuple[dict[str, object], ...],
        str,
        float,
    ]:
        """Run flat pre-solves and retain the cheapest horizontal board.

        The four rotations do not assume a previous calibration.  They cover
        the proper 180-degree sensor-axis conventions that are especially
        ambiguous for a scanner whose measurements all satisfy ``y_S=0``.
        """
        base_rotation = np.asarray(nominal_handeye_rotation, dtype=float)
        translation = np.asarray(nominal_handeye_translation, dtype=float)
        attempts: list[dict[str, object]] = []
        plausible: list[tuple[float, str, float, CalibrationResult]] = []
        converged: list[tuple[float, str, float, CalibrationResult]] = []
        for name, offset in self._flat_rotation_hypotheses():
            try:
                result = self._solve_flat(
                    poses,
                    measurements,
                    base_rotation @ offset,
                    translation,
                    board_dimensions=board_dimensions,
                    initial_board_rotation=initial_board_rotation,
                )
                tilt = self._board_tilt_deg(result.estimate.board.rotation)
                accepted = bool(
                    result.converged
                    and tilt <= self.multistart_maximum_board_tilt_deg
                )
                attempts.append(
                    {
                        "name": name,
                        "converged": bool(result.converged),
                        "cost": float(result.cost),
                        "board_tilt_deg": float(tilt),
                        "accepted": accepted,
                        "message": str(result.message),
                    }
                )
                if result.converged:
                    item = (float(result.cost), name, float(tilt), result)
                    converged.append(item)
                    if accepted:
                        plausible.append(item)
            except Exception as error:
                attempts.append(
                    {
                        "name": name,
                        "converged": False,
                        "cost": None,
                        "board_tilt_deg": None,
                        "accepted": False,
                        "message": str(error),
                    }
                )

        pool = plausible
        if not pool and not self.multistart_require_plausible:
            pool = converged
        if not pool:
            summary = ", ".join(
                (
                    f"{item['name']}:cost={item['cost']},"
                    f"tilt={item['board_tilt_deg']}"
                )
                for item in attempts
            )
            raise RuntimeError(
                "flat multi-start found no converged horizontal-board "
                f"candidate (maximum tilt "
                f"{self.multistart_maximum_board_tilt_deg:.1f} deg): "
                f"{summary}"
            )
        cost, name, tilt, selected = min(pool, key=lambda item: item[0])
        return selected, tuple(attempts), name, tilt

    @staticmethod
    def _attach_initialization_diagnostics(
        result: CalibrationResult,
        attempts: tuple[dict[str, object], ...],
        selected_name: str,
        selected_tilt: float,
        selected_flat_cost: float,
    ) -> CalibrationResult:
        diagnostics = replace(
            result.diagnostics,
            initialization_method="flat_multistart",
            initialization_candidates=attempts,
            selected_initialization=selected_name,
            selected_flat_cost=float(selected_flat_cost),
            selected_board_tilt_deg=float(selected_tilt),
        )
        return replace(result, diagnostics=diagnostics)

    def _solve_flat(
        self,
        poses: list[FlangePose],
        measurements: list[Measurement],
        nominal_handeye_rotation: np.ndarray,
        nominal_handeye_translation: np.ndarray,
        *,
        board_dimensions: tuple[float, float],
        initial_board_rotation: np.ndarray | None = None,
    ) -> CalibrationResult:
        if len(poses) != len(measurements):
            raise ValueError("poses and measurements must have equal length")
        if len(poses) < 4:
            raise ValueError("12-DOF-V2 needs at least four bilateral poses")

        handeye_rotation = np.asarray(nominal_handeye_rotation, dtype=float)
        handeye_translation = np.asarray(nominal_handeye_translation, dtype=float)
        previous_board_rotation = initial_board_rotation
        if initial_board_rotation is None:
            initial_board_rotation = _initial_board_rotation(
                poses, measurements, handeye_rotation, handeye_translation
            )
        x0 = np.concatenate(
            (so3_log(handeye_rotation), handeye_translation, so3_log(initial_board_rotation))
        )
        residual_function = lambda state: variable_projection_residual(
            state, poses, measurements, **self.weights
        )
        optimized = least_squares(
            residual_function,
            x0,
            method="trf",
            x_scale="jac",
            max_nfev=self.max_evaluations,
            ftol=self.tolerance,
            xtol=self.tolerance,
            gtol=self.tolerance,
        )
        state = optimized.x.copy()
        corner, corner_rank = solve_corner(state, poses, measurements, **self.weights)
        if corner_rank < 3:
            raise RuntimeError("projected corner system is rank deficient")

        endpoint_u_offsets: list[np.ndarray] = []
        endpoint_v_offsets: list[np.ndarray] = []
        handeye_rotation_final = so3_exp(state[:3])
        handeye_translation_final = state[3:6]
        for pose, measurement in zip(poses, measurements):
            sensor_rotation = pose.rotation @ handeye_rotation_final
            sensor_translation = pose.translation + pose.rotation @ handeye_translation_final
            endpoint_u_offsets.append(
                sensor_rotation @ measurement.endpoint_u + sensor_translation - corner
            )
            endpoint_v_offsets.append(
                sensor_rotation @ measurement.endpoint_v + sensor_translation - corner
            )
        board_rotation = canonicalize_plane_frame(
            so3_exp(state[6:9]),
            reference=previous_board_rotation,
            endpoint_u_offsets=np.asarray(endpoint_u_offsets),
            endpoint_v_offsets=np.asarray(endpoint_v_offsets),
        )
        state[6:9] = so3_log(board_rotation)
        corner, corner_rank = solve_corner(state, poses, measurements, **self.weights)
        if corner_rank < 3:
            raise RuntimeError("canonicalized corner system is rank deficient")

        residual = residual_function(state)
        jacobian = numerical_jacobian(residual_function, state)
        covariance, residual_variance = covariance_from_jacobian(
            jacobian,
            residual,
            state_scale=self.flat_state_scale,
            fitted_nuisance_parameters=3,
        )
        singular_values, rank, condition = observability(
            jacobian, state_scale=self.flat_state_scale
        )
        effective = effective_handeye_information(
            jacobian, state_scale=self.flat_state_scale
        )
        state_information = (
            scaled_jacobian(jacobian, self.flat_state_scale).T
            @ scaled_jacobian(jacobian, self.flat_state_scale)
        )
        _, _, right_singular = np.linalg.svd(
            scaled_jacobian(jacobian, self.flat_state_scale), full_matrices=False
        )
        board = BoardModel(
            corner=corner,
            rotation=board_rotation,
            length_u=float(board_dimensions[0]),
            length_v=float(board_dimensions[1]),
        )
        estimate = CalibrationEstimate(
            handeye_rotation=so3_exp(state[:3]),
            handeye_translation=state[3:6],
            board=board,
            x9=state,
            covariance_x9=covariance,
            surface_model="flat",
        )
        return CalibrationResult(
            estimate=estimate,
            cost=0.5 * float(residual @ residual),
            converged=bool(
                optimized.success
                and rank == 9
                and condition <= self.maximum_condition_number
            ),
            message=str(optimized.message),
            evaluations=int(optimized.nfev),
            diagnostics=SolverDiagnostics(
                singular_values=singular_values,
                rank=rank,
                condition_number=condition,
                residual_variance=residual_variance,
                effective_handeye_information=effective,
                weakest_direction=right_singular[-1],
                surface_model="flat",
                prior_augmented_singular_values=singular_values,
                prior_augmented_rank=rank,
                prior_augmented_condition_number=condition,
                prior_augmented_effective_handeye_information=effective,
                state_information=state_information,
            ),
        )

    def _solve_shared(
        self,
        poses: list[FlangePose],
        measurements: list[Measurement],
        nominal_handeye_rotation: np.ndarray,
        nominal_handeye_translation: np.ndarray,
        *,
        board_dimensions: tuple[float, float],
        initial_board_rotation: np.ndarray | None,
        initial_estimate: CalibrationEstimate | None,
    ) -> CalibrationResult:
        if len(poses) != len(measurements):
            raise ValueError("poses and measurements must have equal length")
        if len(poses) < 4:
            raise ValueError("12-DOF-V2 needs at least four bilateral poses")
        assert self.surface_basis is not None

        expected_size = 12 + self.surface_basis.size
        if (
            initial_estimate is not None
            and initial_estimate.surface_model == "shared"
            and initial_estimate.state is not None
            and len(initial_estimate.state) == expected_size
            and initial_estimate.surface_basis_kind == self.surface_basis_kind
            and initial_estimate.surface_degree == self.surface_degree
        ):
            initial = initial_estimate.state.copy()
        elif (
            initial_estimate is not None
            and initial_estimate.surface_model == "flat"
        ):
            # Reuse one paired flat initialization when comparing several
            # shape bases on exactly the same dataset.
            initial = np.concatenate(
                (
                    initial_estimate.x9,
                    initial_estimate.board.corner,
                    np.zeros(self.surface_basis.size),
                )
            )
        else:
            flat = self._solve_flat(
                poses,
                measurements,
                nominal_handeye_rotation,
                nominal_handeye_translation,
                board_dimensions=board_dimensions,
                initial_board_rotation=initial_board_rotation,
            )
            estimate = flat.estimate
            initial = np.concatenate(
                (
                    estimate.x9,
                    estimate.board.corner,
                    np.zeros(self.surface_basis.size),
                )
            )

        function = lambda state: self.residual(
            state,
            poses,
            measurements,
            board_dimensions=board_dimensions,
        )
        optimized = least_squares(
            function,
            initial,
            method="trf",
            x_scale=self.state_scale,
            max_nfev=self.max_evaluations,
            ftol=self.tolerance,
            xtol=self.tolerance,
            gtol=self.tolerance,
        )
        state = optimized.x.copy()
        residual = function(state)
        # Re-evaluate centrally so solve, rolling NBV and virtual information
        # use the same numerical linearization convention.
        jacobian = numerical_jacobian(function, state)
        # The regularizer is a prior, not measured evidence.  Its Jacobian can
        # make unsupported shape modes look observable, so convergence and
        # the public rank/condition diagnostics use only observation rows.
        prior_row_count = (
            self.surface_basis.size
            if self.shape_regularization > 0.0
            else 0
        )
        if prior_row_count:
            data_residual = residual[:-prior_row_count]
            data_jacobian = jacobian[:-prior_row_count]
        else:
            data_residual = residual
            data_jacobian = jacobian
        covariance, residual_variance = covariance_from_jacobian(
            jacobian,
            residual,
            state_scale=self.state_scale,
            variance_residual=data_residual,
        )
        data_singular_values, data_rank, data_condition = observability(
            data_jacobian, state_scale=self.state_scale
        )
        prior_singular_values, prior_rank, prior_condition = observability(
            jacobian, state_scale=self.state_scale
        )
        data_effective = effective_handeye_information(
            data_jacobian, state_scale=self.state_scale
        )
        prior_effective = effective_handeye_information(
            jacobian, state_scale=self.state_scale
        )
        jacobian_scaled = scaled_jacobian(jacobian, self.state_scale)
        state_information = jacobian_scaled.T @ jacobian_scaled
        _, _, right_singular = np.linalg.svd(
            scaled_jacobian(data_jacobian, self.state_scale),
            full_matrices=False,
        )
        coefficients = state[12:].copy()
        surface_rms, surface_maximum = self.surface_basis.rms_and_maximum(
            coefficients
        )
        board = BoardModel(
            corner=state[9:12],
            rotation=so3_exp(state[6:9]),
            length_u=float(board_dimensions[0]),
            length_v=float(board_dimensions[1]),
        )
        estimate = CalibrationEstimate(
            handeye_rotation=so3_exp(state[:3]),
            handeye_translation=state[3:6],
            board=board,
            x9=state[:9],
            covariance_x9=covariance[:9, :9],
            state=state,
            covariance_state=covariance,
            surface_model="shared",
            surface_basis_kind=self.surface_basis_kind,
            surface_degree=self.surface_degree,
            shape_coefficients=coefficients,
        )
        return CalibrationResult(
            estimate=estimate,
            cost=0.5 * float(residual @ residual),
            converged=bool(
                optimized.success
                and data_rank == len(state)
                and data_condition <= self.maximum_condition_number
            ),
            message=str(optimized.message),
            evaluations=int(optimized.nfev),
            diagnostics=SolverDiagnostics(
                singular_values=data_singular_values,
                rank=data_rank,
                condition_number=data_condition,
                residual_variance=residual_variance,
                effective_handeye_information=data_effective,
                weakest_direction=right_singular[-1],
                surface_model="shared",
                surface_rms_m=surface_rms,
                surface_maximum_m=surface_maximum,
                prior_augmented_singular_values=prior_singular_values,
                prior_augmented_rank=prior_rank,
                prior_augmented_condition_number=prior_condition,
                prior_augmented_effective_handeye_information=prior_effective,
                state_information=state_information,
            ),
        )
