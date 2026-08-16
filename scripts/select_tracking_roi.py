#!/usr/bin/env python3
"""Interactive tool: pick the first frame and draw the target ROI, save JSON.

This replaces the ROI part of the old visualizer.  The saved JSON is consumed
by benchmark_roi_trackers.py so EVERY tracker starts from the same frame+ROI.

Usage
-----
    python3 select_tracking_roi.py BAG.mcap [--stride 8] [--output roi.json]

Controls
--------
  Left/right : step frame
  drag       : draw ROI (red box) on the profile panel
  s          : save ROI JSON
  q          : quit
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
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
from calibration_pipeline.tools.bag_timeline import BagTimeline


class ROISelector:
    def __init__(self, bag: Path, stride: int = 8, output: Path = Path("tracking_roi.json")):
        self.tl = BagTimeline(bag, stride=stride)
        self.output = output
        self.idx = 0
        self.roi = None
        self.drag_start = None
        self.roi_patch = None

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.ax.set_title("Drag to select target ROI (W_L-E1-P-E2-W_R); s=save, q=quit")
        self.ax.set_xlabel("X [mm]")
        self.ax.set_ylabel("Z [mm]")
        self.ax.grid(True, alpha=0.3)

        self.fig.subplots_adjust(bottom=0.22)
        ax_slider = self.fig.add_axes([0.12, 0.10, 0.72, 0.03])
        self.slider = Slider(ax_slider, "frame", 0, len(self.tl) - 1,
                             valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)

        ax_save = self.fig.add_axes([0.12, 0.04, 0.12, 0.04])
        self.btn_save = Button(ax_save, "Save ROI")
        self.btn_save.on_clicked(self._save)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._render()

    def _render(self) -> None:
        self.ax.cla()
        self.ax.set_title("Drag to select target ROI (W_L-E1-P-E2-W_R); s=save, q=quit")
        self.ax.set_xlabel("X [mm]")
        self.ax.set_ylabel("Z [mm]")
        self.ax.grid(True, alpha=0.3)
        f = self.tl.frames[self.idx]
        prof = f["profile"]
        if len(prof):
            self.ax.scatter(prof[:, 0] * 1000, prof[:, 2] * 1000, s=1.2, c="0.5")
        # bag endpoints as reference if available
        ep = f["endpoints"]
        if ep is not None and len(ep) >= 2:
            self.ax.scatter(ep[:, 0] * 1000, ep[:, 2] * 1000, c="green", s=60, marker="*")
        self.roi_patch = None
        if self.roi is not None:
            x0, z0, x1, z1 = self.roi
            self.roi_patch = Rectangle(
                (min(x0, x1), min(z0, z1)), abs(x1 - x0), abs(z1 - z0),
                fill=False, edgecolor="red", linewidth=2, linestyle="--", zorder=8,
            )
            self.ax.add_patch(self.roi_patch)
        self.ax.set_title(
            f"frame {self.idx}/{len(self.tl)-1} t={self.tl.time_s(f):.1f}s | "
            f"drag=ROI  s=save  q=quit"
        )
        self.fig.canvas.draw_idle()

    def _on_slider(self, val: float) -> None:
        self.idx = int(round(val))
        self._render()

    def _on_press(self, event) -> None:
        if event.inaxes is not self.ax or event.button != 1:
            return
        self.drag_start = (event.xdata, event.ydata)
        self.roi = (event.xdata, event.ydata, event.xdata, event.ydata)
        self._render()

    def _on_motion(self, event) -> None:
        if self.drag_start is None or event.inaxes is not self.ax:
            return
        x0, z0 = self.drag_start
        self.roi = (x0, z0, event.xdata, event.ydata)
        self._render()

    def _on_release(self, event) -> None:
        if self.drag_start is None:
            return
        if event.inaxes is not self.ax:
            return
        x0, z0 = self.drag_start
        self.roi = (x0, z0, event.xdata, event.ydata)
        self.drag_start = None
        self._render()

    def _on_key(self, event) -> None:
        if event.key == "left":
            self.idx = max(0, self.idx - 1)
            self.slider.set_val(self.idx)
        elif event.key == "right":
            self.idx = min(len(self.tl) - 1, self.idx + 1)
            self.slider.set_val(self.idx)
        elif event.key == "s":
            self._save()
        elif event.key == "q":
            plt.close(self.fig)

    def _save(self, _event=None) -> None:
        if self.roi is None:
            print("no ROI drawn")
            return
        x0, z0, x1, z1 = self.roi
        spec = {
            "bag": str(self.tl.path),
            "start_frame": self.idx,
            "roi": {
                "xmin_m": round(min(x0, x1) / 1000.0, 6),
                "zmin_m": round(min(z0, z1) / 1000.0, 6),
                "xmax_m": round(max(x0, x1) / 1000.0, 6),
                "zmax_m": round(max(z0, z1) / 1000.0, 6),
            },
        }
        self.output.write_text(json.dumps(spec, indent=2))
        print(f"saved {self.output}: frame {self.idx} "
              f"roi=({spec['roi']['xmin_m']},{spec['roi']['zmin_m']}).."
              f"({spec['roi']['xmax_m']},{spec['roi']['zmax_m']}) m")

    def show(self) -> None:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--output", type=Path, default=Path("tracking_roi.json"))
    args = ap.parse_args()
    sel = ROISelector(args.bag.resolve(), stride=args.stride, output=args.output)
    print(f"loaded {len(sel.tl)} frames; drag ROI, s=save, q=quit")
    sel.show()


if __name__ == "__main__":
    main()
