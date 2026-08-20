"""Calibration-free translation servos used during seed exploration.

``TranslationServo`` is the original one-feature controller.  It is kept for
repeatability and as an explicitly selectable legacy mode.

``BroydenDualFeatureServo`` controls the measured feature vector
``[x_mid, endpoint_separation]`` with local flange translations
``[dx, dy, dz]``.  Its 2x3 image Jacobian is initialized by small measured
probes and then updated only from accepted, physically valid observations.
"""

from __future__ import annotations

import numpy as np


class BroydenDualFeatureServo:
    """Damped least-squares 2-feature/3-axis servo with Broyden updates.

    All distances are expressed in metres.  Consequently the Jacobian maps
    local flange translation in metres to feature changes in metres and is
    dimensionless.  This avoids the hidden millimetre/metre weighting error
    that would otherwise make the endpoint-separation objective dominate.
    """

    def __init__(
        self,
        *,
        gain: float = 0.55,
        damping: float = 0.05,
        maximum_axis_step: float = 0.003,
        maximum_norm_step: float = 0.005,
        minimum_update_step: float = 0.0002,
        minimum_singular_value: float = 0.02,
        maximum_condition: float = 100.0,
        maximum_model_error_ratio: float = 3.0,
    ) -> None:
        if not 0.0 < gain <= 1.0:
            raise ValueError("gain must lie in (0, 1]")
        if damping < 0.0:
            raise ValueError("damping must be non-negative")
        if maximum_axis_step <= 0.0 or maximum_norm_step <= 0.0:
            raise ValueError("servo step limits must be positive")
        self.gain = float(gain)
        self.damping = float(damping)
        self.maximum_axis_step = float(maximum_axis_step)
        self.maximum_norm_step = float(maximum_norm_step)
        self.minimum_update_step = float(minimum_update_step)
        self.minimum_singular_value = float(minimum_singular_value)
        self.maximum_condition = float(maximum_condition)
        self.maximum_model_error_ratio = float(maximum_model_error_ratio)
        self.jacobian: np.ndarray | None = None
        self.last_model_error_ratio = float("nan")
        self.rejected_update_count = 0
        self.last_update_reason = "not_attempted"

    def set_jacobian(self, jacobian: np.ndarray) -> None:
        value = np.asarray(jacobian, dtype=float)
        if value.shape != (2, 3) or not np.all(np.isfinite(value)):
            raise ValueError("dual-feature Jacobian must be a finite 2x3 matrix")
        self.jacobian = value.copy()
        self.rejected_update_count = 0
        self.last_update_reason = "initialized"

    def health(self, jacobian: np.ndarray | None = None) -> dict[str, object]:
        value = self.jacobian if jacobian is None else np.asarray(jacobian, dtype=float)
        if value is None or value.shape != (2, 3) or not np.all(np.isfinite(value)):
            return {
                "healthy": False,
                "rank": 0,
                "singular_values": [0.0, 0.0],
                "condition": float("inf"),
            }
        singular_values = np.linalg.svd(value, compute_uv=False)
        rank = int(np.sum(singular_values >= self.minimum_singular_value))
        condition = (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > 1e-12
            else float("inf")
        )
        return {
            "healthy": bool(
                rank == 2
                and singular_values[-1] >= self.minimum_singular_value
                and condition <= self.maximum_condition
            ),
            "rank": rank,
            "singular_values": [float(value) for value in singular_values],
            "condition": condition,
        }

    def correction(
        self,
        feature_error: np.ndarray,
        *,
        gain: float | None = None,
        maximum_axis_step: float | None = None,
        maximum_norm_step: float | None = None,
    ) -> np.ndarray:
        """Return a bounded local-flange translation that reduces ``error``."""
        if self.jacobian is None:
            raise RuntimeError("probe a 2x3 Jacobian before requesting a correction")
        health = self.health()
        if not health["healthy"]:
            raise RuntimeError(
                "dual-feature Jacobian is rank deficient or ill-conditioned"
            )
        error = np.asarray(feature_error, dtype=float).reshape(-1)
        if error.shape != (2,) or not np.all(np.isfinite(error)):
            raise ValueError("feature_error must be a finite two-vector")
        regularized = (
            self.jacobian @ self.jacobian.T
            + self.damping**2 * np.eye(2)
        )
        selected_gain = self.gain if gain is None else float(gain)
        axis_limit = (
            self.maximum_axis_step
            if maximum_axis_step is None
            else float(maximum_axis_step)
        )
        norm_limit = (
            self.maximum_norm_step
            if maximum_norm_step is None
            else float(maximum_norm_step)
        )
        if not 0.0 < selected_gain <= 1.0:
            raise ValueError("correction gain must lie in (0, 1]")
        if axis_limit <= 0.0 or norm_limit <= 0.0:
            raise ValueError("correction step limits must be positive")
        step = -selected_gain * self.jacobian.T @ np.linalg.solve(
            regularized, error
        )
        step = np.clip(step, -axis_limit, axis_limit)
        norm = float(np.linalg.norm(step))
        if norm > norm_limit:
            step *= norm_limit / norm
        return step

    def update(
        self,
        actual_local_translation: np.ndarray,
        measured_feature_delta: np.ndarray,
    ) -> bool:
        """Apply a validity-gated good-Broyden update.

        The caller is responsible for accepting only observations that still
        belong to the intended physical edge pair.  An implausible secant is
        rejected without modifying the last known-good Jacobian.
        """
        if self.jacobian is None:
            raise RuntimeError("cannot update an uninitialized Jacobian")
        delta_q = np.asarray(actual_local_translation, dtype=float).reshape(-1)
        delta_s = np.asarray(measured_feature_delta, dtype=float).reshape(-1)
        if (
            delta_q.shape != (3,)
            or delta_s.shape != (2,)
            or not np.all(np.isfinite(delta_q))
            or not np.all(np.isfinite(delta_s))
        ):
            raise ValueError("Broyden update requires finite 3-D/2-D deltas")
        denominator = float(delta_q @ delta_q)
        if denominator < self.minimum_update_step**2:
            # A sub-resolution move carries no useful secant information. It
            # is not evidence that the current model is wrong and therefore
            # must not trigger an expensive local re-probe.
            self.last_update_reason = "step_too_small"
            self.last_model_error_ratio = float("nan")
            self.rejected_update_count = 0
            return False
        prediction_residual = delta_s - self.jacobian @ delta_q
        self.last_model_error_ratio = float(
            np.linalg.norm(prediction_residual)
            / max(np.linalg.norm(delta_s), self.minimum_update_step)
        )
        if self.last_model_error_ratio > self.maximum_model_error_ratio:
            self.rejected_update_count += 1
            self.last_update_reason = "model_error"
            return False
        proposed = self.jacobian + np.outer(
            prediction_residual, delta_q
        ) / denominator
        if not self.health(proposed)["healthy"]:
            self.rejected_update_count += 1
            self.last_update_reason = "unhealthy_proposal"
            return False
        self.jacobian = proposed
        self.rejected_update_count = 0
        self.last_update_reason = "accepted"
        return True


class TranslationServo:
    def __init__(
        self,
        *,
        gain: float = 0.8,
        maximum_step: float = 0.015,
        sensitivity_smoothing: float = 0.5,
        minimum_sensitivity: float = 1e-4,
    ) -> None:
        self.gain = gain
        self.maximum_step = maximum_step
        self.sensitivity_smoothing = sensitivity_smoothing
        self.minimum_sensitivity = minimum_sensitivity
        self.axis: int | None = None
        self.sensitivity: float | None = None

    def choose_axis(self, sensitivities: dict[int, float]) -> int:
        usable = {
            axis: value
            for axis, value in sensitivities.items()
            if np.isfinite(value) and abs(value) >= self.minimum_sensitivity
        }
        if not usable:
            raise ValueError("all translation probe sensitivities are negligible")
        self.axis = max(usable, key=lambda axis: abs(usable[axis]))
        self.sensitivity = float(usable[self.axis])
        return self.axis

    def update(self, delta_x_mid: float, delta_translation: float) -> float:
        if abs(delta_translation) < 1e-12:
            raise ValueError("delta_translation must be non-zero")
        measured = float(delta_x_mid / delta_translation)
        if abs(measured) < self.minimum_sensitivity:
            return self.sensitivity if self.sensitivity is not None else measured
        if self.sensitivity is None:
            self.sensitivity = measured
        else:
            eta = self.sensitivity_smoothing
            self.sensitivity = (1.0 - eta) * self.sensitivity + eta * measured
        return self.sensitivity

    def correction(self, x_mid: float) -> float:
        if self.axis is None or self.sensitivity is None:
            raise RuntimeError("probe and choose an axis before requesting a correction")
        if abs(self.sensitivity) < self.minimum_sensitivity:
            raise RuntimeError("translation sensitivity is too small")
        step = -self.gain * float(x_mid) / self.sensitivity
        return float(np.clip(step, -self.maximum_step, self.maximum_step))
