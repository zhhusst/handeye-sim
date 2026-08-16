#!/usr/bin/env python3
"""Unified ROI tracker benchmark.

Runs every tracker on the SAME bag, SAME first frame and SAME initial ROI,
then reports:

  - Endpoint containment rate   C_k = 1 if bag-recorded E1/E2 inside ROI_k
  - First failure frame         (tracker update failed OR containment broken)
  - Mean runtime per frame

Usage
-----
    python3 benchmark_roi_trackers.py --bag BAG.mcap --roi-json roi.json
                                      [--trackers csrt,kcf,...]
                                      [--stride 1] [--max-frames N]
                                      [--output out.json]

ROI JSON (produced by select_tracking_roi.py):
    {"bag": "...", "start_frame": 0, "roi": {"xmin_m":..., ...}}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/scripts")
from calibration_pipeline.tools.bag_timeline import BagTimeline
from calibration_pipeline.roi_tracking.factory import create_tracker, TRACKER_NAMES
from calibration_pipeline.roi_tracking.rasterizer import (
    ProfileRasterizer,
    RasterizerConfig,
)
from calibration_pipeline.roi_tracking.types import TrackingFrame, TargetROI, ROITrackingResult


def rasterizer_for_profiles(profiles, margin_m=0.02, res_m_per_px=0.00025):
    """Fixed X-Z canvas covering all profile points + margin (same for all frames)."""
    xs, zs = [], []
    for prof in profiles:
        if len(prof):
            xs.append(prof[:, 0].min()); xs.append(prof[:, 0].max())
            zs.append(prof[:, 2].min()); zs.append(prof[:, 2].max())
    return ProfileRasterizer(RasterizerConfig(
        x_min_m=min(xs) - margin_m,
        x_max_m=max(xs) + margin_m,
        z_min_m=min(zs) - margin_m,
        z_max_m=max(zs) + margin_m,
        resolution_m_per_pixel=res_m_per_px,
        point_radius_px=2,
    ))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--roi-json", type=Path, required=True,
                    help="JSON with start_frame + roi (select_tracking_roi.py)")
    ap.add_argument("--trackers", default=",".join(TRACKER_NAMES),
                    help="comma-separated tracker names")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--res-mm", type=float, default=0.25,
                    help="raster resolution in mm per pixel")
    args = ap.parse_args()

    spec = json.loads(args.roi_json.read_text())
    start_frame = int(spec.get("start_frame", 0))
    roi = TargetROI.from_dict(spec["roi"])
    tracker_names = [t.strip().lower() for t in args.trackers.split(",") if t.strip()]

    tl = BagTimeline(args.bag.resolve(), stride=args.stride)
    print(f"bag frames: {len(tl)} (stride {args.stride})")

    rasterizer = rasterizer_for_profiles(
        [f["profile"] for f in tl.frames], res_m_per_px=args.res_mm / 1000.0
    )
    print(f"raster canvas: {rasterizer.config.width_px} x {rasterizer.config.height_px} px, "
          f"{args.res_mm} mm/px")

    # build TrackingFrames lazily (rasterise on demand)
    def make_frame(i):
        f = tl.frames[i]
        return TrackingFrame(
            timestamp_s=tl.time_s(f),
            profile=f["profile"],
            image=rasterizer.rasterize(f["profile"]),
        )

    end = len(tl) if args.max_frames is None else min(start_frame + args.max_frames, len(tl))
    print(f"evaluating frames {start_frame}..{end-1}\n")

    report = {"bag": str(args.bag), "start_frame": start_frame,
              "roi": roi.to_dict(), "stride": args.stride,
              "trackers": {}}

    frame0 = make_frame(start_frame)
    for name in tracker_names:
        t0 = time.perf_counter()
        tracker = create_tracker(name, rasterizer)
        try:
            tracker.initialize(frame0, roi)
        except Exception as exc:
            print(f"[{name}] init FAILED: {exc}")
            report["trackers"][name] = {"init_error": str(exc)}
            continue
        init_ms = (time.perf_counter() - t0) * 1000.0

        contained = 0
        total = 0
        unevaluable = 0
        first_failure = None
        runtime_list = []
        roi_traj = []
        for i in range(start_frame + 1, end):
            fr = make_frame(i)
            try:
                res = tracker.update(fr)
            except Exception as exc:
                res = ROITrackingResult(False, None, None, 0.0, f"exception:{exc}")
            runtime_list.append(res.runtime_ms)
            ep = tl.frames[i]["endpoints"]
            if ep is None or len(ep) < 2:
                # no ground-truth this frame (node was LOST) -> unrateable,
                # NOT a failure
                unevaluable += 1
                if res.roi is not None:
                    roi_traj.append({"frame": i, "success": res.success,
                                     "roi": res.roi.to_dict(),
                                     "reason": res.reason,
                                     "runtime_ms": round(res.runtime_ms, 3)})
                continue
            inside = bool(
                res.success
                and res.roi is not None
                and np.all(res.roi.contains(ep[:, (0, 2)]))
            )
            total += 1
            if res.success and inside:
                contained += 1
            if first_failure is None and not (res.success and inside):
                first_failure = i
            if res.roi is not None:
                roi_traj.append({"frame": i, "success": res.success,
                                 "roi": res.roi.to_dict(),
                                 "reason": res.reason,
                                 "runtime_ms": round(res.runtime_ms, 3)})
        contain_rate = contained / total if total else 0.0
        mean_runtime = float(np.mean(runtime_list)) if runtime_list else 0.0
        report["trackers"][name] = {
            "containment_rate": round(contain_rate, 4),
            "contained_frames": contained,
            "total_frames": total,
            "unevaluable_frames": unevaluable,
            "first_failure_frame": first_failure,
            "mean_runtime_ms": round(mean_runtime, 3),
            "init_ms": round(init_ms, 3),
            "trajectory": roi_traj,
        }
        print(f"[{name:22s}] containment={contain_rate*100:6.1f}%  "
              f"({contained}/{total} rated, {unevaluable} skipped)  "
              f"first_failure={first_failure}  runtime={mean_runtime:.1f}ms")

    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
