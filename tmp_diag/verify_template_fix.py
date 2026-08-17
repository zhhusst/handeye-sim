#!/usr/bin/env python3
"""离线验证 handle_prior 修复: 候选端点更新模板后, 模板方向与候选一致。"""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/ros2_ws/src/fanuc_m20id25_support")
from calibration_pipeline.nbv.candidate_generator import generate_candidates
from calibration_pipeline.models import BoardModel, CalibrationEstimate
from calibration_pipeline.geometry import make_transform
from calibration_pipeline.roi_tracking.breakpoint_pipeline import (
    ROIBreakpointPipeline,
    ROIBreakpointPipelineConfig,
)

def so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-10:
        return np.zeros(3)
    return 0.5 * a / np.sin(a) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

result = json.load(open("/workspace/data/calibration_runs/20260817_live2/calibration_result.json"))
R_he = np.array(result["handeye"]["rotation"])
t_he = np.array(result["handeye"]["translation"])
corner = np.array(result["board"]["corner"])
R_board = np.array(result["board"]["rotation"])
u_hat = R_board[:, 0]
seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]

board = BoardModel(corner=corner, rotation=R_board, length_u=0.2, length_v=0.15)
x9 = np.concatenate([so3_log(R_he), t_he, so3_log(R_board)])
estimate = CalibrationEstimate(handeye_rotation=R_he, handeye_translation=t_he, board=board, x9=x9)

R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)
sensor_ref = T_fb @ T_he

cands = generate_candidates(estimate, edge_samples=4, edge_margin=0.04,
    alphas_deg=(25.0,31.0,38.0), psis_deg=(-15.0,0.0,15.0),
    working_distances=(0.24,0.28,0.33), profile_samples=40,
    reference_sensor_transform=sensor_ref)
c369 = [c for c in cands if c.candidate_id == "candidate_00369"][0]
T_s = c369.sensor_transform_nominal
R_s = T_s[:3, :3]; t_s = T_s[:3, 3]
p_u = corner + c369.a * u_hat
p_v = corner + c369.b * u_hat
u_pred = R_s.T @ (p_u - t_s)
v_pred = R_s.T @ (p_v - t_s)
prior = np.vstack((u_pred, v_pred))

# 创建 pipeline
config = ROIBreakpointPipelineConfig()
pipe = ROIBreakpointPipeline(config)

print("修复前模板端点:")
t0 = pipe.template_endpoints
print("  ", np.round(t0, 4))
ang0 = np.degrees(np.arctan2(t0[1,2]-t0[0,2], t0[1,0]-t0[0,0]))
print("  模板连线角: %.1f°" % ang0)

print("\n调用 handle_prior(候选端点):")
pipe.handle_prior(prior)
print("修复后模板端点:")
t1 = pipe.template_endpoints
print("  ", np.round(t1, 4))
ang1 = np.degrees(np.arctan2(t1[1,2]-t1[0,2], t1[1,0]-t1[0,0]))
print("  模板连线角: %.1f° (候选 %.1f°)" % (ang1, np.degrees(np.arctan2(prior[1,2]-prior[0,2], prior[1,0]-prior[0,0]))))
print("  模式:", pipe.mode)
print("  guide:", np.round(pipe.guide_endpoints, 3))
