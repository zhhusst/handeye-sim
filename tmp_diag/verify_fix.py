#!/usr/bin/env python3
"""验证: 修复后, 用参考位姿的参数生成的候选, 关节差是否≈0。

如果修复正确: 用参考的 (alpha=31.3, dist=0.28, 扫描角≈-62°) 生成候选,
flange 应与参考 flange 几乎一致 => IK 关节差小。
"""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/ros2_ws/src/fanuc_m20id25_support")
from calibration_pipeline.nbv.candidate_generator import generate_candidates
from calibration_pipeline.models import BoardModel, CalibrationEstimate
from calibration_pipeline.geometry import make_transform
from fanuc_m20id25_support.fanuc_kinematic import inverse_kinematics_numeric

def so3_log(R):
    cos_ang = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(cos_ang)
    if ang < 1e-10:
        return np.zeros(3)
    return 0.5 * ang / np.sin(ang) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

result = json.load(open("/workspace/data/calibration_runs/20260817_live2/calibration_result.json"))
R_he = np.array(result["handeye"]["rotation"])
t_he = np.array(result["handeye"]["translation"])
corner = np.array(result["board"]["corner"])
R_board = np.array(result["board"]["rotation"])

seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]
joints_ref = np.array(ref["joints"])
R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)
sensor_ref = T_fb @ T_he

board = BoardModel(corner=corner, rotation=R_board, length_u=0.2, length_v=0.15)
x9 = np.concatenate([so3_log(R_he), t_he, so3_log(R_board)])
estimate = CalibrationEstimate(handeye_rotation=R_he, handeye_translation=t_he, board=board, x9=x9)

# 用参考参数
candidates = generate_candidates(
    estimate,
    edge_samples=4, edge_margin=0.04,
    alphas_deg=(31.3,),
    psis_deg=(-15.0, 0.0, 15.0),
    working_distances=(0.28,),
    profile_samples=40,
)
print(f"候选: {len(candidates)}")
for c in candidates:
    d_sensor = np.linalg.norm(c.sensor_transform_nominal[:3,3] - sensor_ref[:3,3])
    sols = inverse_kinematics_numeric(c.flange_transform_command, q_init=joints_ref)
    if len(sols) == 0:
        continue
    q = sols[0]
    delta = np.degrees(np.max(np.abs(q - joints_ref)))
    print(f"  branch={c.branch:+d} a={c.a:.3f} b={c.b:.3f} psi={np.degrees(c.psi):.0f}° "
          f"sensor_dist={d_sensor:.3f} joint_delta={delta:.1f}°")
