#!/usr/bin/env python3
"""Interactive bag visualizer for breakpoint tracking diagnosis.

Shows, frame by frame, what the robot is doing (joint angles), what the laser
profile looks like, where the tracker's guide is, where the detected endpoints
are, and exactly when/how tracking is lost.  The user scrubs through the bag,
observes, and reports what they see; this drives root-cause discussion.

Usage
-----
    python3 visualize_tracking_bag.py BAG.mcap [--stride 4] [--portrait]

Controls
--------
  - Left/right arrows:  step one frame
  - Space:              play/pause
  - [ ]:                slow down / speed up playback
  - Home/End:           jump to start/end
  - Mouse wheel:        fine scrub (on slider)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
from calibration_pipeline.perception.tracking_pipeline import (
    BreakpointTrackingPipeline,
    TrackingPipelineConfig,
)
from calibration_pipeline.perception.endpoint_detector import EndpointDetectionConfig
from real_config import make_real_config


# ---------------------------------------------------------------------------
# Bag loading
# ---------------------------------------------------------------------------
def _pc2_points(message) -> np.ndarray:
    import sensor_msgs_py.point_cloud2 as point_cloud2

    values = point_cloud2.read_points(
        message, field_names=("x", "y", "z"), skip_nans=True
    )
    if len(values) == 0:
        return np.zeros((0, 3), dtype=float)
    if getattr(values.dtype, "names", None):
        return np.column_stack(
            tuple(np.asarray(values[name], dtype=float) for name in ("x", "y", "z"))
        )
    return np.asarray(values, dtype=float).reshape(-1, 3)


class BagTimeline:
    """Load a bag into a time-synchronized frame list."""

    def __init__(self, path: Path, stride: int = 1) -> None:
        self.path = path
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""),
        )
        self.types = {i.name: i.type for i in reader.get_all_topics_and_types()}
        self.profile_type = get_message(self.types.get("/gocator/profile", "sensor_msgs/msg/PointCloud2"))
        self.joint_type = get_message(self.types.get("/joint_states", "sensor_msgs/msg/JointState"))
        self.flange_type = get_message(self.types.get("/calibration/flange_pose", "geometry_msgs/msg/PoseStamped"))
        self.guide_type = get_message(self.types.get("/calibration/detection_guide", "sensor_msgs/msg/PointCloud2"))
        self.endpoint_type = get_message(self.types.get("/calibration/endpoints", "sensor_msgs/msg/PointCloud2"))
        self.diag_type = get_message(self.types.get("/profile_endpoint_detector/diagnostics", "std_msgs/msg/String"))

        # Raw streams (time-stamped)
        self.profiles: list[tuple[int, np.ndarray]] = []
        self.joints: list[tuple[int, np.ndarray]] = []
        self.flanges: list[tuple[int, np.ndarray]] = []
        self.guides: list[tuple[int, np.ndarray]] = []
        self.endpoints: list[tuple[int, np.ndarray]] = []
        self.diags: list[tuple[int, dict]] = []

        while reader.has_next():
            topic, data, ts = reader.read_next()
            if topic == "/gocator/profile":
                self.profiles.append((ts, _pc2_points(deserialize_message(data, self.profile_type))))
            elif topic == "/joint_states":
                msg = deserialize_message(data, self.joint_type)
                self.joints.append((ts, np.asarray(msg.position, dtype=float)))
            elif topic == "/calibration/flange_pose":
                msg = deserialize_message(data, self.flange_type)
                p = msg.pose.position
                q = msg.pose.orientation
                self.flanges.append((ts, np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=float)))
            elif topic == "/calibration/detection_guide":
                self.guides.append((ts, _pc2_points(deserialize_message(data, self.guide_type))))
            elif topic == "/calibration/endpoints":
                self.endpoints.append((ts, _pc2_points(deserialize_message(data, self.endpoint_type))))
            elif topic == "/profile_endpoint_detector/diagnostics":
                msg = deserialize_message(data, self.diag_type)
                try:
                    self.diags.append((ts, json.loads(msg.data)))
                except json.JSONDecodeError:
                    pass

        self.stride = max(1, stride)
        # Build frame list: every stride-th profile, synced with nearest others
        self.frames: list[dict] = []
        for idx, (ts, prof) in enumerate(self.profiles):
            if idx % self.stride:
                continue
            frame = {
                "stamp": ts,
                "profile": prof,
                "joints": self._nearest(self.joints, ts),
                "flange": self._nearest(self.flanges, ts),
                "guide": self._nearest(self.guides, ts),
                "endpoints": self._nearest(self.endpoints, ts),
                "diag": self._nearest(self.diags, ts),
            }
            self.frames.append(frame)
        self.t0 = self.frames[0]["stamp"] if self.frames else 0

    @staticmethod
    def _nearest(stream, ts):
        if not stream:
            return None
        times = np.asarray([s[0] for s in stream], dtype=np.int64)
        i = int(np.searchsorted(times, ts))
        if i >= len(stream):
            i = len(stream) - 1
        return stream[i][1]

    def __len__(self) -> int:
        return len(self.frames)

    def time_s(self, frame: dict) -> float:
        return (frame["stamp"] - self.t0) * 1e-9


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------
class TrackingVisualizer:
    def __init__(self, timeline: BagTimeline, replay_pipeline: bool = True, initial_mode: str = "TRACK") -> None:
        self.tl = timeline
        self.idx = 0
        self.playing = False
        self.play_interval_ms = 100
        self.replay_pipeline = replay_pipeline
        self.initial_mode = initial_mode
        # manual ROI state (mm, in profile axes)
        self.roi = None            # (x0, z0, x1, z1) in mm
        self.roi_drag_start = None
        self.roi_patch = None
        self.roi_locked = False
        self.roi_detected = None   # endpoints found inside ROI (m, 2x3)
        self.pipeline = None
        if replay_pipeline:
            # Single parameter source shared with replay_tracking_bag.py
            # (real-machine detector_parameters.yaml).  P1-5.
            config = make_real_config()
            config.initial_mode = self.initial_mode
            self.pipeline = BreakpointTrackingPipeline(config)

        # ---- live pipeline cache ----
        # history[i] = (FrameResult|None, diag dict); None = not computed yet.
        # Frames are computed on demand in order (like the real node), so
        # playback is live; scrubbing shows cached frames instantly.
        self.history: list[tuple | None] = [None] * len(self.tl)

        # ---- figure layout ----
        self.fig = plt.figure(figsize=(16, 9))
        gs = self.fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[1.3, 1, 1])
        self.ax_profile = self.fig.add_subplot(gs[0, 0])
        self.ax_status = self.fig.add_subplot(gs[0, 1])
        self.ax_joints = self.fig.add_subplot(gs[0, 2])
        self.ax_len = self.fig.add_subplot(gs[1, 0])
        self.ax_traj = self.fig.add_subplot(gs[1, 1])
        self.ax_summary = self.fig.add_subplot(gs[1, 2])

        self.ax_status.axis("off")
        self._init_axes()

        # ---- controls ----
        self.fig.subplots_adjust(bottom=0.22)
        ax_slider = self.fig.add_axes([0.12, 0.10, 0.72, 0.03])
        self.slider = Slider(
            ax_slider, "frame", 0, max(len(self.tl) - 1, 1),
            valinit=0, valstep=1,
        )
        self.slider.on_changed(self._on_slider)

        ax_play = self.fig.add_axes([0.12, 0.04, 0.12, 0.04])
        self.btn_play = Button(ax_play, "Play ▶")
        self.btn_play.on_clicked(self._toggle_play)

        ax_reset = self.fig.add_axes([0.26, 0.04, 0.12, 0.04])
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_reset.on_clicked(self._reset)

        ax_roi = self.fig.add_axes([0.40, 0.04, 0.14, 0.04])
        self.btn_roi = Button(ax_roi, "Lock ROI")
        self.btn_roi.on_clicked(self._apply_roi_lock)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self._timer = self.fig.canvas.new_timer(interval=self.play_interval_ms)
        self._timer.add_callback(self._tick)

        self._render()

    # -- axis setup ---------------------------------------------------------
    def _init_axes(self) -> None:
        self.ax_profile.set_title("Laser profile (X-Z, mm)")
        self.ax_profile.set_xlabel("X [mm]")
        self.ax_profile.set_ylabel("Z [mm]")
        self.ax_profile.grid(True, alpha=0.3)

        self.ax_joints.set_title("Joint angles [deg]")
        self.ax_joints.set_xlabel("time [s]")
        self.ax_joints.grid(True, alpha=0.3)

        self.ax_len.set_title("Tracked chord length [mm]")
        self.ax_len.set_xlabel("time [s]")
        self.ax_len.grid(True, alpha=0.3)

        self.ax_traj.set_title("Endpoint trajectory (X, Z)")
        self.ax_traj.set_xlabel("X [mm]")
        self.ax_traj.set_ylabel("Z [mm]")
        self.ax_traj.grid(True, alpha=0.3)

        self.ax_summary.set_title("Rejections / loss timeline")
        self.ax_summary.set_xlabel("time [s]")
        self.ax_summary.grid(True, alpha=0.3)

    # -- rendering ----------------------------------------------------------
    def _ensure_computed(self, up_to: int) -> None:
        """Compute frames up to ``up_to`` in strict order, caching each result.
        The pipeline is a state machine: every frame must be processed exactly
        once, in order, so we always continue from the first uncomputed slot.
        This makes playback live (frame-by-frame) while scrubbing backwards
        stays instant and state-consistent."""
        if self.pipeline is None:
            for i in range(len(self.history)):
                if self.history[i] is None:
                    self.history[i] = (None, self.tl.frames[i]["diag"] or {})
            return
        start = 0
        while start < len(self.history) and self.history[start] is not None:
            start += 1
        if start > up_to:
            return  # already computed
        for i in range(start, min(up_to + 1, len(self.tl))):
            f = self.tl.frames[i]
            r = self.pipeline.process_profile(f["profile"], timestamp_s=self.tl.time_s(f))
            self.history[i] = (r, r.to_dict())

    def _render(self) -> None:
        self._ensure_computed(self.idx)
        f = self.tl.frames[self.idx]
        t = self.tl.time_s(f)
        result, diag = self.history[self.idx]
        self.ax_profile.cla()
        self.ax_joints.cla()
        self.ax_len.cla()
        self.ax_traj.cla()
        self.ax_summary.cla()
        self._init_axes()

        # profile points (raw bag data, the only bag content shown)
        prof = f["profile"]
        if prof is not None and len(prof):
            x, z = prof[:, 0] * 1000, prof[:, 2] * 1000
            self.ax_profile.scatter(x, z, s=1.2, c="0.5", label="profile")

        # manual ROI overlay.  ax_profile.cla() wipes artists each frame, so
        # the rectangle is re-created every render.
        self.roi_patch = None
        if self.roi is not None:
            from matplotlib.patches import Rectangle
            x0, z0, x1, z1 = self.roi
            self.roi_patch = Rectangle(
                (min(x0, x1), min(z0, z1)),
                abs(x1 - x0), abs(z1 - z0),
                fill=False, edgecolor="red", linewidth=2,
                linestyle="--", zorder=8,
            )
            self.ax_profile.add_patch(self.roi_patch)

        # pipeline output only: detected endpoints (orange x) + guide (cyan ^)
        if result is not None:
            if result.endpoints is not None:
                ex = result.endpoints[:, 0] * 1000
                ez = result.endpoints[:, 2] * 1000
                self.ax_profile.scatter(ex, ez, c="orange", s=90, marker="x",
                                        zorder=7, label="pipeline endpoints")
            g1 = np.array(result.guide_first_mm)
            g2 = np.array(result.guide_second_mm)
            self.ax_profile.scatter([g1[0]], [g1[2]], c="cyan", s=50, marker="^", zorder=6)
            self.ax_profile.scatter([g2[0]], [g2[2]], c="cyan", s=50, marker="^", zorder=6)
            self.ax_profile.plot([g1[0], g2[0]], [g1[2], g2[2]], "c--", linewidth=1, zorder=5)

        # status text (pipeline diagnostics)
        self.ax_status.cla()
        self.ax_status.axis("off")
        mode = diag.get("mode", "?")
        state = diag.get("state", "?")
        reason = diag.get("reason", "")
        lines = [
            f"frame {self.idx}/{len(self.tl)-1}",
            f"t = {t:.2f} s",
            f"mode = {mode}",
            f"state = {state}",
            f"reason = {reason}",
            f"accepted = {diag.get('accepted', '?')} / {diag.get('frames', '?')}",
            f"acceptance = {diag.get('acceptance_rate', '?')}",
            f"lost_frames = {diag.get('lost_frames', '?')}",
            f"reacquire = {diag.get('reacquire_frames', '?')}",
            f"seg_len = {diag.get('segment_length_mm', '?')}",
            f"temporal = {diag.get('temporal_tracking', '?')}",
            f"temporal_susp = {diag.get('temporal_suspended', '?')}",
            f"reflen = {diag.get('tracking_reference_length_mm', '?')}",
            f"roi = {'locked' if self.roi_locked else 'none'} (drag to set, Lock ROI to start)",
        ]
        self.ax_status.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
                            family="monospace", fontsize=9,
                            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

        # joints history (bag joint angles: what the robot is doing)
        times = [self.tl.time_s(ff) for ff in self.tl.frames[: self.idx + 1]]
        joint_hist = []
        for ff in self.tl.frames[: self.idx + 1]:
            if ff["joints"] is not None:
                joint_hist.append(ff["joints"])
        if joint_hist and times:
            J = np.asarray(joint_hist)
            for j in range(min(J.shape[1], 6)):
                self.ax_joints.plot(times, np.rad2deg(J[:, j]), label=f"J{j+1}")
        self.ax_joints.legend(fontsize=7, ncol=2, loc="upper right") if joint_hist else None

        # chord length from pipeline endpoints
        lengths, ltimes = [], []
        for i in range(self.idx + 1):
            rr, _ = self.history[i]
            if rr is not None and rr.endpoints is not None:
                lengths.append(np.linalg.norm(rr.endpoints[1] - rr.endpoints[0]) * 1000)
                ltimes.append(self.tl.time_s(self.tl.frames[i]))
        if lengths:
            self.ax_len.plot(ltimes, lengths, "-", color="orange")
            self.ax_len.axhline(np.median(lengths), ls="--", color="gray", alpha=0.6)

        # endpoint trajectory from pipeline
        ep_hist = [
            (self.tl.time_s(self.tl.frames[i]), rr.endpoints)
            for i in range(self.idx + 1)
            if self.history[i][0] is not None and self.history[i][0].endpoints is not None
        ]
        if ep_hist:
            x1 = [e[0, 0] * 1000 for _, e in ep_hist]
            z1 = [e[0, 2] * 1000 for _, e in ep_hist]
            x2 = [e[1, 0] * 1000 for _, e in ep_hist]
            z2 = [e[1, 2] * 1000 for _, e in ep_hist]
            self.ax_traj.plot(x1, z1, "-", color="orange", alpha=0.7, label="e1")
            self.ax_traj.plot(x2, z2, "-", color="blue", alpha=0.7, label="e2")
        if result is not None:
            g1 = np.array(result.guide_first_mm)
            g2 = np.array(result.guide_second_mm)
            self.ax_traj.scatter([g1[0], g2[0]], [g1[2], g2[2]], c="cyan", s=40, zorder=5)
        self.ax_traj.legend(fontsize=7, loc="upper right") if ep_hist else None

        # state timeline from pipeline
        for i in range(self.idx + 1):
            rr, dd = self.history[i]
            tt = self.tl.time_s(self.tl.frames[i])
            st = dd.get("state")
            if st == "LOST":
                self.ax_summary.scatter(tt, 1, c="red", s=12)
            elif st == "REJECTED":
                self.ax_summary.scatter(tt, 0, c="orange", s=8)
            elif st == "VALID":
                self.ax_summary.scatter(tt, -0.2, c="green", s=4)
        self.ax_summary.set_ylim(-0.6, 1.6)
        self.ax_summary.set_yticks([-0.2, 0, 1])
        self.ax_summary.set_yticklabels(["VALID", "REJECT", "LOST"], fontsize=7)

        self.ax_profile.legend(fontsize=7, loc="lower right")
        self.fig.canvas.draw_idle()


    def _on_press(self, event) -> None:
        if event.inaxes is not self.ax_profile:
            return
        if event.button != 1:
            return
        self.roi_drag_start = (event.xdata, event.ydata)
        self.roi = (event.xdata, event.ydata, event.xdata, event.ydata)
        self._render()

    def _on_motion(self, event) -> None:
        if self.roi_drag_start is None:
            return
        if event.inaxes is not self.ax_profile:
            return
        x0, z0 = self.roi_drag_start
        self.roi = (x0, z0, event.xdata, event.ydata)
        self._render()

    def _on_release(self, event) -> None:
        if self.roi_drag_start is None:
            return
        if event.inaxes is not self.ax_profile:
            return
        x0, z0 = self.roi_drag_start
        self.roi = (x0, z0, event.xdata, event.ydata)
        self.roi_drag_start = None
        self._render()

    def _apply_roi_lock(self, _event=None) -> None:
        """Use the drawn ROI as the initial guide, then enter TRACK."""
        if self.pipeline is None or self.roi is None:
            print("draw an ROI first (drag on the profile panel)")
            return
        x0, z0, x1, z1 = self.roi
        xlo, xhi = min(x0, x1), max(x0, x1)
        zlo, zhi = min(z0, z1), max(z0, z1)
        if xhi - xlo < 5 or zhi - zlo < 2:
            print("ROI too small")
            return

        # STEP 1: detect the two breakpoints INSIDE the ROI rectangle.
        # A temporary guide runs along the configured plate angle through the
        # ROI centre; its corridor is sized to cover the whole ROI rectangle
        # (normal gate = half the ROI height, endpoint gate = half the ROI
        # width), so detection happens inside the drawn box -- not a narrow
        # diagonal band.  If the first attempt fails, the corridor widens
        # progressively; as a last resort the ROI geometry itself is used.
        f = self.tl.frames[self.idx]
        prof = f["profile"]
        angle = np.deg2rad(self.pipeline.config.alignment_template_angle_deg)
        ux, uz = float(np.cos(angle)), float(np.sin(angle))
        cx = 0.5 * (xlo + xhi) / 1000.0
        cz = 0.5 * (zlo + zhi) / 1000.0
        half = 0.5 * (xhi - xlo) / 1000.0 / max(abs(ux), 1e-6)
        temp_guide = np.array([
            [cx - half * ux, 0.0, cz - half * uz],
            [cx + half * ux, 0.0, cz + half * uz],
        ])
        normal_gate = 0.5 * (zhi - zlo) / 1000.0
        endpoint_gate = 0.5 * (xhi - xlo) / 1000.0
        guide = None
        for scale in (1.0, 1.5, 2.0, 3.0):
            det = self.pipeline.detector.detect_guided(
                prof,
                temp_guide[0],
                temp_guide[1],
                normal_gate_m=normal_gate * scale,
                endpoint_gate_m=endpoint_gate * scale,
                maximum_angle_difference_deg=(
                    self.pipeline.config.tracking_maximum_angle_difference_deg
                ),
                selection_mode="guided_track",
            )
            if det is None:
                continue
            candidate = np.asarray(
                np.vstack((det.first, det.second)), dtype=float
            ).reshape(2, 3)
            # verify both endpoints lie inside (or within 10 mm of) the ROI
            margin = 0.010
            inside = bool(
                np.all(candidate[:, 0] >= (xlo / 1000.0 - margin))
                and np.all(candidate[:, 0] <= (xhi / 1000.0 + margin))
                and np.all(candidate[:, 2] >= (zlo / 1000.0 - margin))
                and np.all(candidate[:, 2] <= (zhi / 1000.0 + margin))
            )
            if inside:
                guide = candidate
                break
        if guide is None:
            # P1-8: never fabricate endpoints from the box geometry.  A
            # benchmark must know that tracking started from a REAL detection;
            # a virtual guide would make the first loss untraceable.
            self.btn_roi.label.set_text("Init failed - redraw")
            print("ROI initialization failed: no breakpoints detected inside "
                  "the box; redraw the ROI")
            return
        guide = np.asarray(guide, dtype=float).reshape(2, 3)

        # reset pipeline and initialize TRACK from the ROI guide.  The ROI is a
        # rough prior: leave identity uninitialised so the first TRACK frame
        # re-anchors guide onto the measured endpoints (first-frame skip in
        # _tracking_pair_is_plausible).  A tight 3 mm step gate against the
        # raw ROI would reject any user-drawn box.
        self.pipeline.reset()
        self.pipeline.mode = "TRACK"
        self.pipeline.guide_endpoints = guide.copy()
        self.pipeline.tracking_expected = guide.copy()
        self.pipeline.identity_initialized = False
        self.pipeline.lost_frames = 0
        self.pipeline.reacquire_frames = 0
        # P0-2: without SEED_TRACK_START the temporal (Kalman) tracker is
        # never requested and the GUI would silently benchmark the plain
        # previous-frame-guide loop instead of the real tracking algorithm.
        self.pipeline.handle_control("SEED_TRACK_START")
        self.roi_locked = True
        self.btn_roi.label.set_text("ROI locked")
        print(
            f"ROI detect: endpoints "
            f"{np.round(guide[0]*1000,1).tolist()} "
            f"{np.round(guide[1]*1000,1).tolist()} "
            f"len={np.linalg.norm(guide[1]-guide[0])*1000:.1f}mm"
        )
        # drop cached results from the current frame onward; the pipeline was
        # reset+initialised with the ROI guide, so the next _render recomputes
        # the current frame live and playback continues from there.
        for i in range(self.idx, len(self.history)):
            self.history[i] = None
        self._render()

    # -- interaction --------------------------------------------------------
    def _on_slider(self, val: float) -> None:
        self.idx = int(round(val))
        self._render()

    def _toggle_play(self, _event=None) -> None:
        self.playing = not self.playing
        if self.playing:
            self.btn_play.label.set_text("Pause ⏸")
            self._timer.start()
        else:
            self.btn_play.label.set_text("Play ▶")
            self._timer.stop()

    def _reset(self, _event=None) -> None:
        self.playing = False
        self.btn_play.label.set_text("Play ▶")
        self._timer.stop()
        self.roi = None
        self.roi_drag_start = None
        self.roi_locked = False
        self.roi_detected = None
        self.btn_roi.label.set_text("Lock ROI")
        if self.roi_patch is not None:
            self.roi_patch.remove()
            self.roi_patch = None
        self.idx = 0
        self.slider.set_val(0)
        self.history = [None] * len(self.tl)
        if self.pipeline is not None:
            self.pipeline.reset()
        self._render()

    def _tick(self) -> None:
        if self.idx < len(self.tl) - 1:
            self.idx += 1
            self.slider.set_val(self.idx)
        else:
            self.playing = False
            self.btn_play.label.set_text("Play ▶")
            self._timer.stop()

    def _on_key(self, event) -> None:
        if event.key == "left":
            self.idx = max(0, self.idx - 1)
            self.slider.set_val(self.idx)
        elif event.key == "right":
            self.idx = min(len(self.tl) - 1, self.idx + 1)
            self.slider.set_val(self.idx)
        elif event.key == " ":
            self._toggle_play()
        elif event.key == "home":
            self.idx = 0
            self.slider.set_val(0)
        elif event.key == "end":
            self.idx = len(self.tl) - 1
            self.slider.set_val(self.idx)
        elif event.key == "]":
            self.play_interval_ms = max(20, self.play_interval_ms - 20)
            self._timer.interval = self.play_interval_ms
        elif event.key == "[":
            self.play_interval_ms = min(1000, self.play_interval_ms + 20)
            self._timer.interval = self.play_interval_ms

    def _on_close(self, _event=None) -> None:
        try:
            self._timer.stop()
        except Exception:
            pass

    def show(self) -> None:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--stride", type=int, default=1,
                        help="process every Nth profile (default 1 = real frame rate)")
    parser.add_argument("--no-pipeline", action="store_true",
                        help="disable the offline pipeline overlay (bag diagnostics only)")
    parser.add_argument("--mode", choices=["align", "track"], default="track",
                        help="offline pipeline initial mode (default track)")
    args = parser.parse_args()

    tl = BagTimeline(args.bag.resolve(), stride=args.stride)
    print(f"Loaded {len(tl)} frames from {args.bag}")
    viz = TrackingVisualizer(
        tl,
        replay_pipeline=not args.no_pipeline,
        initial_mode=args.mode.upper(),
    )
    viz.show()


if __name__ == "__main__":
    main()
