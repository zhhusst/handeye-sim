#!/usr/bin/env python3
"""Headless replay of the online ROI/CSRT breakpoint pipeline on an MCAP bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from calibration_pipeline.roi_tracking import (
    ROIBreakpointPipeline,
    ROIBreakpointPipelineConfig,
)
from calibration_pipeline.tools.bag_timeline import BagTimeline


def replay(path: Path, *, stride: int, maximum_frames: int | None) -> dict:
    timeline = BagTimeline(path, stride=stride)
    pipeline = ROIBreakpointPipeline(
        ROIBreakpointPipelineConfig(
            initial_first_label="e2",
            alignment_template_center_x_m=0.0,
            alignment_template_center_z_m=0.28,
            alignment_template_length_m=0.08,
            alignment_template_angle_deg=25.0,
            minimum_lock_frames=5,
            minimum_segment_length_m=0.010,
            maximum_segment_length_m=0.25,
        )
    )
    states: dict[str, int] = {}
    rows = []
    locked_frame = None
    previous_mode = pipeline.mode
    lost_episodes = 0
    recovered_episodes = 0
    consecutive_misses = 0
    maximum_consecutive_misses = 0
    started = time.perf_counter()
    for index, frame in enumerate(timeline.frames):
        if maximum_frames is not None and index >= maximum_frames:
            break
        result = pipeline.process_profile(
            frame["profile"], timeline.time_s(frame)
        )
        if locked_frame is None and pipeline.alignment_stable_frames >= 5:
            if pipeline.lock():
                locked_frame = index
        if previous_mode != "LOST" and pipeline.mode == "LOST":
            lost_episodes += 1
        if previous_mode == "LOST" and pipeline.mode != "LOST":
            recovered_episodes += 1
        previous_mode = pipeline.mode
        states[result.state] = states.get(result.state, 0) + 1
        if result.state == "VALID":
            consecutive_misses = 0
        else:
            consecutive_misses += 1
            maximum_consecutive_misses = max(
                maximum_consecutive_misses, consecutive_misses
            )
        rows.append(
            {
                "frame": index,
                "time_s": timeline.time_s(frame),
                "state": result.state,
                "mode": result.mode,
                "reason": result.reason,
                "segment_length_mm": result.segment_length_mm,
            }
        )
    elapsed = time.perf_counter() - started
    valid = states.get("VALID", 0)
    return {
        "bag": str(path),
        "backend": "roi_csrt",
        "stride": stride,
        "frames": len(rows),
        "locked_frame": locked_frame,
        "states": states,
        "acceptance_rate": valid / max(len(rows), 1),
        "lost_episodes": lost_episodes,
        "recovered_episodes": recovered_episodes,
        "maximum_consecutive_misses": maximum_consecutive_misses,
        "processing_time_s": elapsed,
        "mean_processing_ms": 1000.0 * elapsed / max(len(rows), 1),
        "final_mode": pipeline.mode,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--maximum-frames", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    result = replay(
        args.bag.resolve(),
        stride=args.stride,
        maximum_frames=args.maximum_frames,
    )
    summary = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

