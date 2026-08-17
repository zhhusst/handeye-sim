#!/usr/bin/env python3
"""验证: branch=+1 正面候选 vs 参考位姿的关节差分布。

关键: 如果 branch=+1 候选存在关节差 < 70° 的, 则只需过滤 branch=-1
或让候选生成默认正面。如果连 branch=+1 都超 70°, 是更深的问题。
"""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/ros2_ws/src/fanuc_m20id25_support")
from calibration_pipeline.nbv.candidate_generator import generate_candidates
from calibration_pipeline.models import BoardModel, CalibrationEstimate
from fanuc_m20id25_support.fanuc_kinematic import forward_kinematics_urdf, inverse_kinematics_numeric

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
T_fb = np.eye(4); T_fb[:3,:3] = R_f; T_fb[:3,3] = t_f

board = BoardModel(corner=corner, rotation=R_board, length_u=0.2, length_v=0.15)
x9 = np.concatenate([so3_log(R_he), t_he, so3_log(R_board)])
estimate = CalibrationEstimate(handeye_rotation=R_he, handeye_translation=t_he, board=board, x9=x9)

for branch in (1, -1):
    candidates = generate_candidates(
        estimate,
        edge_samples=4, edge_margin=0.04,
        alphas_deg=(20.0, 35.0, 50.0),
        psis_deg=(-15.0, 0.0, 15.0),
        working_distances=(0.33, 0.4, 0.5),
        profile_samples=40,
    )
    # 只保留当前 branch
    cands = [c for c in candidates if c.branch == branch]
    deltas = []
    ik_ok = 0
    for c in cands:
        sols = inverse_kinematics_numeric(c.flange_transform_command, q_init=joints_ref)
        if len(sols) == 0:
            continue
        ik_ok += 1
        q = sols[0]
        delta = np.degrees(np.max(np.abs(q - joints_ref)))
        deltas.append(delta)
    deltas = np.array(deltas)
    print(f"branch={branch:+d}: 候选 {len(cands)}, IK有解 {ik_ok}, "
          f"关节差 max min={deltas.min():.1f}° med={np.median(deltas):.1f}° "
          f"<70°: {(deltas<70).sum()}")

# 也测: 更保守的参数 (贴参考: 小 alpha, 小距离)
print("\n=== 贴参考参数 (alpha=31°≈参考入射角, dist=0.24≈参考距离) ===")
for branch in (1, -1):
    candidates = generate_candidates(
        estimate,
        edge_samples=4, edge_margin=0.04,
        alphas_deg=(31.0,),
        psis_deg=(0.0,),
        working_distances=(0.24,),
        profile_samples=40,
    )
    cands = [c for c in candidates if c.branch == branch]
    deltas = []
    for c in cands:
        sols = inverse_kinematics_numeric(c.flange_transform_command, q_init=joints_ref)
        if len(sols) == 0:
            continue
        q = sols[0]
        delta = np.degrees(np.max(np.abs(q - joints_ref)))
        deltas.append(delta)
    deltas = np.array(deltas)
    print(f"branch={branch:+d}: 候选 {len(cands)}, IK有解 {len(deltas)}, "
          f"min={deltas.min():.1f}° med={np.median(deltas):.1f}° <70°: {(deltas<70).sum()}")
