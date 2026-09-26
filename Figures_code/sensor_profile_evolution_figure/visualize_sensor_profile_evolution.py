#!/usr/bin/env python3
"""Publication figure for real 2-D profiles measured in the sensor frame.

The default A/B/C profiles are the same three consecutive states used by
``rotation_servo_microstep_figure``:

* A -- valid dual-edge observation before orientation excitation;
* B -- after a pure +1 degree local-y_F rotation and before translation servo;
* C -- after the immediately following translation-servo correction.

Only the cached, bag-extracted measurements are read.  The script does not
start ROS and does not modify any calibration or robot-control code.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STATES = (
    SCRIPT_DIR.parent
    / "rotation_servo_microstep_figure"
    / "microstep_states.json"
)

STATE_COLORS = {
    "A": "#4D4D4D",  # neutral reference
    "B": "#D55E00",  # Okabe-Ito vermillion
    "C": "#0072B2",  # Okabe-Ito blue
}
STATE_LINESTYLES = {"A": "-", "B": "-", "C": "-"}
PANEL_TITLES = {
    "A": r"(a) Valid observation before rotation",
    "B": r"(b) After $+1^{\circ}$ local-$y_F$ rotation",
    "C": r"(c) After translation-servo correction",
}
SHORT_NAMES = {
    "A": "A  Before rotation",
    "B": r"B  After $+1^{\circ}$ rotation",
    "C": "C  After translation servo",
}


@dataclass(frozen=True)
class ProfileState:
    key: str
    profile_mm: np.ndarray
    endpoints_mm: np.ndarray
    x_mid_mm: float
    separation_mm: float


@dataclass(frozen=True)
class ProfileSequence:
    source_run: str
    source_bag: str
    branch: str
    states: tuple[ProfileState, ...]
    metrics: dict
    control_target: dict


def configure_publication_style() -> str:
    """Prefer Times New Roman and use its metric-compatible local fallback."""
    import matplotlib as mpl
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    if "Times New Roman" in available:
        selected = "Times New Roman"
    elif "Nimbus Roman" in available:
        # The container does not ship Microsoft's proprietary font. Nimbus
        # Roman has Times-compatible proportions and is preferable to a sans
        # fallback. If Times New Roman is installed later, it is selected
        # automatically without changing this script.
        selected = "Nimbus Roman"
    elif "Liberation Serif" in available:
        selected = "Liberation Serif"
    else:
        selected = "serif"

    mpl.rcParams.update(
        {
            "font.family": selected,
            "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": True,
            "axes.linewidth": 0.85,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Keep text editable in SVG authoring software.
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    return selected


def _finite_array(value, *, columns: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != columns or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite N x {columns} array")
    return array


def load_sequence(path: Path) -> ProfileSequence:
    payload = json.loads(path.read_text(encoding="utf-8"))
    states: list[ProfileState] = []
    for item in payload["states"]:
        key = str(item["key"])
        profile_m = _finite_array(
            item["target_surface_profile"]["points_S_m"],
            columns=3,
            name=f"{key}.profile",
        )
        endpoints_m = _finite_array(
            item["endpoints_S_m"], columns=3, name=f"{key}.endpoints"
        )
        if endpoints_m.shape != (2, 3):
            raise ValueError(f"{key}.endpoints must contain exactly two points")
        # Gocator profile points are a 2-D x-z cross section. Sorting by x
        # avoids drawing acquisition-order zigzags without changing any point.
        order = np.argsort(profile_m[:, 0], kind="stable")
        states.append(
            ProfileState(
                key=key,
                profile_mm=1000.0 * profile_m[order],
                endpoints_mm=1000.0 * endpoints_m,
                x_mid_mm=float(item["x_mid_mm"]),
                separation_mm=float(item["endpoint_separation_mm"]),
            )
        )
    expected = ("A", "B", "C")
    if tuple(state.key for state in states) != expected:
        raise ValueError(f"expected state order {expected}, got {[s.key for s in states]}")
    return ProfileSequence(
        source_run=str(payload["source_run"]),
        source_bag=str(payload["source_bag"]),
        branch=str(payload["branch"]),
        states=tuple(states),
        metrics=dict(payload["metrics"]),
        control_target=dict(payload["control_target"]),
    )


def common_limits(sequence: ProfileSequence) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.vstack(
        [
            np.vstack((state.profile_mm[:, [0, 2]], state.endpoints_mm[:, [0, 2]]))
            for state in sequence.states
        ]
    )
    x_min, z_min = np.min(points, axis=0)
    x_max, z_max = np.max(points, axis=0)
    x_pad = max(5.0, 0.075 * (x_max - x_min))
    z_pad = max(3.0, 0.10 * (z_max - z_min))
    return (x_min - x_pad, x_max + x_pad), (z_min - z_pad, z_max + z_pad)


def _draw_profile(
    ax,
    state: ProfileState,
    *,
    show_midline: bool = True,
    annotate_endpoints: bool = True,
    line_alpha: float = 1.0,
    line_width: float = 1.75,
    marker_size: float = 7.0,
    zorder: int = 4,
) -> None:
    import matplotlib.patheffects as path_effects

    x = state.profile_mm[:, 0]
    z = state.profile_mm[:, 2]
    color = STATE_COLORS[state.key]

    # A pale halo keeps overlapping profiles readable in the overlay version.
    (line,) = ax.plot(
        x,
        z,
        color=color,
        linestyle=STATE_LINESTYLES[state.key],
        linewidth=line_width,
        alpha=line_alpha,
        solid_capstyle="round",
        zorder=zorder,
    )
    line.set_path_effects(
        [path_effects.Stroke(linewidth=line_width + 1.8, foreground="white", alpha=0.75), path_effects.Normal()]
    )
    ax.scatter(
        x,
        z,
        s=marker_size,
        color=color,
        alpha=0.34 * line_alpha,
        edgecolors="none",
        rasterized=True,
        zorder=zorder - 1,
    )

    endpoints = state.endpoints_mm[:, [0, 2]]
    ax.plot(
        endpoints[:, 0],
        endpoints[:, 1],
        linestyle=(0, (3.0, 2.2)),
        color=color,
        linewidth=1.0,
        alpha=0.75 * line_alpha,
        zorder=zorder,
    )
    markers = ("o", "s")
    labels = (r"$e_u$", r"$e_v$")
    for index, (point, marker, endpoint_label) in enumerate(
        zip(endpoints, markers, labels)
    ):
        ax.scatter(
            point[0],
            point[1],
            s=52,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=zorder + 3,
        )
        ax.scatter(
            point[0],
            point[1],
            s=54,
            marker=marker,
            facecolor="none",
            edgecolor=color,
            linewidth=0.8,
            zorder=zorder + 4,
        )
        if annotate_endpoints:
            x_offset = 2.0 if index == 0 else -2.0
            alignment = "left" if index == 0 else "right"
            ax.annotate(
                endpoint_label,
                xy=point,
                xytext=(x_offset, 3.0),
                textcoords="offset points",
                ha=alignment,
                va="bottom",
                color=color,
                fontsize=9.0,
                fontstyle="italic",
                zorder=zorder + 5,
            )

    if show_midline:
        ax.axvline(
            state.x_mid_mm,
            color=color,
            linewidth=0.9,
            linestyle=(0, (1.5, 2.3)),
            alpha=0.65,
            zorder=1,
        )


def _style_axis(ax, x_lim, z_lim, *, show_y_label: bool, equal: bool = True) -> None:
    ax.set_xlim(*x_lim)
    ax.set_ylim(*z_lim)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_S$ (mm)", labelpad=3)
    if show_y_label:
        ax.set_ylabel(r"$z_S$ (mm)", labelpad=4)
    ax.minorticks_on()
    ax.tick_params(which="major", direction="in", length=4.0, width=0.8, top=True, right=True)
    ax.tick_params(which="minor", direction="in", length=2.2, width=0.55, top=True, right=True)
    ax.grid(True, which="major", color="#D7DEE5", linewidth=0.55, alpha=0.62)
    ax.grid(False, which="minor")
    for spine in ax.spines.values():
        spine.set_color("#37474F")
        spine.set_linewidth(0.8)
    ax.set_facecolor("#FCFDFE")


def _metrics_box(ax, state: ProfileState) -> None:
    text = (
        rf"$x_{{\mathrm{{mid}}}}={state.x_mid_mm:+.2f}\ \mathrm{{mm}}$"
        "\n"
        rf"$L={state.separation_mm:.2f}\ \mathrm{{mm}}$"
    )
    ax.text(
        0.035,
        0.945,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color="#263238",
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "white",
            "edgecolor": STATE_COLORS[state.key],
            "linewidth": 0.85,
            "alpha": 0.92,
        },
        zorder=20,
    )


def _figure_transition_annotations(figure, sequence: ProfileSequence) -> None:
    from matplotlib.patches import FancyArrowPatch

    rotation = float(sequence.metrics["A_to_B_rotation_deg"])
    translation = float(sequence.metrics["B_to_C_translation_norm_mm"])
    residual_rotation = float(sequence.metrics["B_to_C_rotation_deg"])
    transitions = (
        (
            0.315,
            0.545,
            rf"Orientation excitation: $+{rotation:.2f}^{{\circ}}$ about local $y_F$",
        ),
        (
            0.650,
            0.880,
            rf"Translation servo: $\|\Delta q_F\|={translation:.2f}$ mm; "
            rf"$\Delta R={residual_rotation:.5f}^{{\circ}}$",
        ),
    )
    for x_start, x_end, label in transitions:
        arrow = FancyArrowPatch(
            (x_start, 0.918),
            (x_end, 0.918),
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#566573",
            clip_on=False,
        )
        figure.add_artist(arrow)
        figure.text(
            0.5 * (x_start + x_end),
            0.932,
            label,
            ha="center",
            va="bottom",
            fontsize=9.4,
            color="#263238",
        )


def draw_triptych(sequence: ProfileSequence, *, title: bool):
    import matplotlib.pyplot as plt

    x_lim, z_lim = common_limits(sequence)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(11.25, 3.95),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.075},
    )
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.15, top=0.83)

    for index, (ax, state) in enumerate(zip(axes, sequence.states)):
        _draw_profile(ax, state)
        _style_axis(ax, x_lim, z_lim, show_y_label=index == 0)
        ax.set_title(PANEL_TITLES[state.key], pad=8.0, color="#1F2D3A")
        _metrics_box(ax, state)
        if index > 0:
            ax.tick_params(labelleft=False)

    _figure_transition_annotations(figure, sequence)
    if title:
        figure.suptitle(
            r"Evolution of the measured plate profile in the sensor frame $\{S\}$",
            x=0.5,
            y=0.995,
            fontsize=12.0,
            fontweight="bold",
        )
    return figure


def draw_overlay(sequence: ProfileSequence, *, title: bool):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    x_lim, z_lim = common_limits(sequence)
    figure, ax = plt.subplots(figsize=(5.65, 4.45))
    figure.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.90)
    for index, state in enumerate(sequence.states):
        _draw_profile(
            ax,
            state,
            show_midline=False,
            annotate_endpoints=False,
            line_alpha=0.94,
            line_width=1.8,
            marker_size=5.0,
            zorder=4 + 3 * index,
        )
    _style_axis(ax, x_lim, z_lim, show_y_label=True)
    handles = [
        Line2D(
            [0],
            [0],
            color=STATE_COLORS[state.key],
            linewidth=2.2,
            marker="o",
            markersize=4.0,
            label=(
                f"{state.key}: "
                + {
                    "A": "before rotation",
                    "B": r"after $+1^{\circ}$ rotation",
                    "C": "after translation servo",
                }[state.key]
                + rf"  ($L={state.separation_mm:.1f}$ mm)"
            ),
        )
        for state in sequence.states
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        borderpad=0.55,
        handlelength=2.0,
    )
    legend.get_frame().set_edgecolor("#9AA7B2")
    legend.get_frame().set_linewidth(0.7)
    if title:
        ax.set_title(
            r"Measured plate-profile evolution in $\{S\}$",
            pad=9.0,
            fontweight="bold",
        )
    return figure


def save_all_formats(figure, output_stem: Path, dpi: int) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in (".pdf", ".svg", ".png"):
        path = output_stem.with_suffix(suffix)
        options = {"dpi": dpi} if suffix == ".png" else {}
        figure.savefig(path, bbox_inches="tight", pad_inches=0.035, **options)
        outputs.append(path)
    return outputs


def write_metadata(
    path: Path,
    sequence: ProfileSequence,
    *,
    selected_font: str,
    source_states: Path,
) -> None:
    payload = {
        "source_states": str(source_states.resolve()),
        "source_run": sequence.source_run,
        "source_bag": sequence.source_bag,
        "branch": sequence.branch,
        "rendered_font": selected_font,
        "states": [
            {
                "key": state.key,
                "profile_point_count": int(len(state.profile_mm)),
                "x_mid_mm": state.x_mid_mm,
                "endpoint_separation_mm": state.separation_mm,
                "endpoint_u_xz_mm": state.endpoints_mm[0, [0, 2]].tolist(),
                "endpoint_v_xz_mm": state.endpoints_mm[1, [0, 2]].tolist(),
            }
            for state in sequence.states
        ],
        "transitions": {
            "A_to_B_rotation_deg": sequence.metrics["A_to_B_rotation_deg"],
            "A_to_B_translation_norm_mm": sequence.metrics[
                "A_to_B_translation_norm_mm"
            ],
            "B_to_C_rotation_deg": sequence.metrics["B_to_C_rotation_deg"],
            "B_to_C_translation_norm_mm": sequence.metrics[
                "B_to_C_translation_norm_mm"
            ],
            "B_to_C_translation_in_B_flange_mm": sequence.metrics[
                "B_to_C_translation_in_B_flange_mm"
            ],
        },
        "note": (
            "Profiles and endpoints are actual target_surface_points and "
            "seed_motion_state measurements extracted from the real bag; no "
            "synthetic profile or geometric refit is drawn."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR, help="Output directory."
    )
    parser.add_argument(
        "--layout",
        choices=("triptych", "overlay", "both"),
        default="both",
        help="Figure layout to export (default: both).",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the overall figure title for journal caption-based layouts.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the generated figure window after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_font = configure_publication_style()
    sequence = load_sequence(args.states)
    figures = []
    generated: list[Path] = []
    if args.layout in {"triptych", "both"}:
        triptych = draw_triptych(sequence, title=not args.no_title)
        figures.append(triptych)
        generated.extend(
            save_all_formats(
                triptych,
                args.output_dir / "sensor_profile_evolution_triptych",
                args.dpi,
            )
        )
    if args.layout in {"overlay", "both"}:
        overlay = draw_overlay(sequence, title=not args.no_title)
        figures.append(overlay)
        generated.extend(
            save_all_formats(
                overlay,
                args.output_dir / "sensor_profile_evolution_overlay",
                args.dpi,
            )
        )
    metadata_path = args.output_dir / "sensor_profile_evolution_metadata.json"
    write_metadata(
        metadata_path,
        sequence,
        selected_font=selected_font,
        source_states=args.states,
    )
    generated.append(metadata_path)

    print(f"Font: {selected_font}")
    print("Generated:")
    for path in generated:
        print(f"  {path}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()
