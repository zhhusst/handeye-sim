#!/usr/bin/env python3
"""Interactive viewer for RECORDED ROI tracking data (no re-tracking).

Shows exactly what the system recorded in the bag:
  - laser profile (mm, sensor frame)
  - ROI1 red / ROI2 blue boxes from diagnostics.roi_boxes_mm
  - endpoints (gold stars)
  - failure reasons text + red background on failed frames
  - mode / state / fail_streak / segment_length / chord limits

ANNOTATION MODE (ground-truth breakpoint marking):
  - press '1' then click canvas  -> mark TRUE E1 at click position (green)
  - press '2' then click canvas  -> mark TRUE E2 at click position (green)
  - press 'd'                    -> delete this frame's annotation
  - press 'n'                    -> append a text note to this frame (terminal)
  - press 's'                    -> save annotations to JSON
Annotations persist across runs via --annotations FILE (or auto path).

Usage:
    python3 recorded_roi_viewer.py BAG.mcap [--stride 4] [--annotations FILE]

Keys:  left/right = one frame, space = play/pause
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
from calibration_pipeline.tools.bag_timeline import BagTimeline


def default_annotation_path(bag: Path) -> Path:
    return bag.with_suffix(bag.suffix + ".annotations.json")


class RecordedROIViewer:
    def __init__(self, bag: Path, stride: int = 4,
                 annotation_file: Path | None = None) -> None:
        self.tl = BagTimeline(bag, stride=stride)
        self.bag = bag
        self.idx = 0
        self.playing = False
        self.annotation_file = annotation_file or default_annotation_path(bag)
        self.annotations: dict[str, dict] = {}
        self._load_annotations()
        self.annot_mode: str | None = None   # None | "e1" | "e2"

        self.fig, self.ax = plt.subplots(figsize=(13, 6.5))
        self.fig.subplots_adjust(bottom=0.24)
        ax_slider = self.fig.add_axes([0.12, 0.12, 0.72, 0.03])
        self.slider = Slider(ax_slider, "frame", 0, len(self.tl) - 1,
                             valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)
        ax_play = self.fig.add_axes([0.12, 0.05, 0.12, 0.05])
        self.btn_play = Button(ax_play, "Play / Pause (space)")
        self.btn_play.on_clicked(self._toggle_play)
        ax_info = self.fig.add_axes([0.28, 0.05, 0.68, 0.05])
        ax_info.axis("off")
        self.info_text = ax_info.text(0.0, 0.5, "", va="center", fontsize=9,
                                      family="monospace")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self._tick)
        self._has_limits = False
        self._render()

    # -- annotations -----------------------------------------------------
    def _load_annotations(self) -> None:
        path = self.annotation_file
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.annotations = payload.get("annotations", {})
            print(f"[viewer] loaded {len(self.annotations)} annotated frames "
                  f"from {path}")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"[viewer] could not load annotations: {error}")

    def save_annotations(self) -> None:
        payload = {
            "bag": self.bag.name,
            "stride": self.tl.stride,
            "frame_count": len(self.tl),
            "annotations": self.annotations,
        }
        path = self.annotation_file
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"[viewer] saved {len(self.annotations)} annotated frames "
              f"to {path}")

    # -- data ----------------------------------------------------------
    def _diag(self, frame) -> dict | None:
        return frame.get("diag")

    def _roi_boxes_mm(self, diag) -> list:
        if not diag:
            return []
        boxes = diag.get("roi_boxes_mm") or []
        out = []
        for b in boxes:
            try:
                out.append((b["xmin_mm"], b["zmin_mm"], b["xmax_mm"], b["zmax_mm"]))
            except (KeyError, TypeError):
                continue
        return out

    # -- render ---------------------------------------------------------
    def _render(self) -> None:
        # Preserve the user's zoom across re-renders (e.g. annotation mode
        # toggles).  The first render autoscales to the full profile.
        prev_xlim = self.ax.get_xlim() if self._has_limits else None
        prev_ylim = self.ax.get_ylim() if self._has_limits else None

        frame = self.tl.frames[self.idx]
        diag = self._diag(frame)
        prof = frame["profile"]
        eps = frame["endpoints"]
        t_s = self.tl.time_s(frame)

        self.ax.clear()
        if prev_xlim is not None:
            self.ax.set_xlim(prev_xlim)
            self.ax.set_ylim(prev_ylim)
        self._has_limits = True
        if prof is not None and len(prof):
            self.ax.scatter(1000.0 * prof[:, 0], 1000.0 * prof[:, 2],
                            s=1.5, c="tab:blue", linewidths=0)

        failed = bool(diag and (diag.get("tracker_failures") or []))
        if failed:
            self.ax.set_facecolor((1.0, 0.85, 0.85))

        colors = ["red", "blue"]
        for i, (x0, z0, x1, z1) in enumerate(self._roi_boxes_mm(diag)):
            col = colors[i % 2]
            self.ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0,
                                        fill=False, edgecolor=col, lw=2))
            self.ax.text(x0, z1 + 1.5, f"ROI{i + 1}", color=col, fontsize=9,
                         fontweight="bold")

        if eps is not None and len(eps):
            self.ax.scatter(1000.0 * eps[:, 0], 1000.0 * eps[:, 2],
                            marker="*", s=220, c="gold", edgecolors="black",
                            zorder=5)

        # annotation overlay
        key = str(self.idx)
        if key in self.annotations:
            ann = self.annotations[key]
            for name, pt in (("e1", ann.get("e1")), ("e2", ann.get("e2"))):
                if pt is None:
                    continue
                x, z = float(pt[0]), float(pt[1])
                self.ax.scatter([x], [z], marker="o", s=180, facecolors="none",
                                edgecolors="green", linewidths=2.5, zorder=6)
                self.ax.text(x + 1.5, z + 1.5, f"TRUE {name.upper()}",
                             color="green", fontsize=9, fontweight="bold")
            note = ann.get("note")
            if note:
                self.ax.text(0.02, 0.98, f"note: {note}",
                             transform=self.ax.transAxes, va="top",
                             color="green", fontsize=9)

        ref = diag.get("tracking_reference_length_mm") if diag else None
        seg = diag.get("segment_length_mm") if diag else None
        lo = 0.45 * ref if ref else None
        hi = 2.5 * ref if ref else None

        info = []
        if diag:
            info.append(f"mode={diag.get('mode')} state={diag.get('state')} "
                        f"fail_streak={diag.get('fail_streak')}/{diag.get('fail_streak_frames')}")
            info.append(f"seg={seg:.1f}mm" if seg is not None else "seg=?")
            if ref:
                info.append(f"chord=[{lo:.1f},{hi:.1f}]mm")
            fails = diag.get("tracker_failures") or []
            if fails:
                info.append("FAIL: " + ",".join(fails))
            else:
                info.append("OK")
        if self.annot_mode:
            info.append(f">>> click TRUE {self.annot_mode.upper()} <<<")
        n_ann = len(self.annotations)
        info.append(f"annot={n_ann}")
        info.append(f"t={t_s:.2f}s frame={self.idx}/{len(self.tl) - 1}")
        self.info_text.set_text(" | ".join(info))

        self.ax.set_xlabel("x (mm)")
        self.ax.set_ylabel("z (mm)")
        self.ax.set_title(f"frame {self.idx} — {self.tl.path.name}")
        self.fig.canvas.draw_idle()

    # -- interactions ---------------------------------------------------
    def _on_slider(self, val):
        self.idx = int(round(val))
        self._render()

    def _on_click(self, event):
        if event.inaxes is not self.ax or self.annot_mode is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        key = str(self.idx)
        self.annotations.setdefault(key, {})[self.annot_mode] = [
            float(event.xdata), float(event.ydata),
        ]
        print(f"[viewer] frame {self.idx}: TRUE {self.annot_mode.upper()} = "
              f"({event.xdata:.2f}, {event.ydata:.2f}) mm")
        self.annot_mode = None
        self._render()

    def _on_key(self, event):
        if event.key == "right":
            self.idx = min(self.idx + 1, len(self.tl) - 1)
            self.slider.set_val(self.idx)
        elif event.key == "left":
            self.idx = max(self.idx - 1, 0)
            self.slider.set_val(self.idx)
        elif event.key == " ":
            self._toggle_play(None)
        elif event.key == "1":
            self.annot_mode = "e1"
            print("[viewer] annotation mode: click TRUE E1")
            self._render()
        elif event.key == "2":
            self.annot_mode = "e2"
            print("[viewer] annotation mode: click TRUE E2")
            self._render()
        elif event.key == "d":
            key = str(self.idx)
            if key in self.annotations:
                del self.annotations[key]
                print(f"[viewer] deleted annotation for frame {self.idx}")
                self._render()
        elif event.key == "n":
            key = str(self.idx)
            note = input(f"note for frame {self.idx}: ").strip()
            if note:
                self.annotations.setdefault(key, {})["note"] = note
                print(f"[viewer] note saved: {note}")
                self._render()
        elif event.key == "s":
            self.save_annotations()

    def _toggle_play(self, _event=None):
        self.playing = not self.playing
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _tick(self):
        if not self.playing:
            return
        if self.idx >= len(self.tl) - 1:
            self.playing = False
            self.timer.stop()
            return
        self.idx += 1
        self.slider.set_val(self.idx)

    def show(self):
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--annotations", type=Path, default=None)
    args = ap.parse_args()
    viewer = RecordedROIViewer(args.bag, stride=args.stride,
                               annotation_file=args.annotations)
    viewer.show()


if __name__ == "__main__":
    main()
