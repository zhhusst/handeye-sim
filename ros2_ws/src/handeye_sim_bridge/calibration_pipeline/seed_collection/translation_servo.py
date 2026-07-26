"""One-dimensional, calibration-free translation servo."""

from __future__ import annotations

import numpy as np


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
