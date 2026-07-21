#!/usr/bin/env python3
"""
collection/io.py — 数据持久化: 保存/加载标定数据

支持格式: recorded_poses.json (与现有 auto_calib_v2_node 兼容)
"""

import json
import os
import numpy as np
from handeye_sim.core.types import Pose, Measurement, CalibRecord, SceneGT, CalibData


def save_calib_data(data: CalibData, filepath: str):
    """保存标定数据到 JSON

    格式与 recorded_poses.json 兼容:
    {
      "poses": [{R_i, t_i, scan_pts_S, p_S_e1, p_S_e2, valid_e1, valid_e2}, ...],
      "scene": {R_he_gt, t_he_gt}
    }
    """
    poses_json = []
    for rec in data.records:
        entry = {
            'R_i': rec.pose.R.tolist(),
            't_i': rec.pose.t.tolist(),
            'scan_pts_S': [p.tolist() for p in rec.meas.scan_pts_S],
            'valid_e1': rec.meas.valid_e1,
            'valid_e2': rec.meas.valid_e2,
        }
        if rec.meas.p_S_e1 is not None:
            entry['p_S_e1'] = rec.meas.p_S_e1.tolist()
        if rec.meas.p_S_e2 is not None:
            entry['p_S_e2'] = rec.meas.p_S_e2.tolist()
        poses_json.append(entry)

    output = {'poses': poses_json}

    if data.scene_gt is not None:
        output['scene'] = {
            'R_he_gt': data.scene_gt.R_he.tolist(),
            't_he_gt': data.scene_gt.t_he.tolist(),
        }

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(data.records)} poses → {filepath}")


def load_calib_data(filepath: str) -> CalibData:
    """从 JSON 加载标定数据"""
    with open(filepath) as f:
        raw = json.load(f)

    records = []
    for p in raw['poses']:
        R_i = np.array(p.get('R_i', p.get('R')))
        t_i = np.array(p.get('t_i', p.get('t')))
        pose = Pose(R=R_i, t=t_i)

        meas = Measurement(
            valid_e1=p.get('valid_e1', False),
            valid_e2=p.get('valid_e2', False),
        )
        if p.get('p_S_e1') is not None:
            meas.p_S_e1 = np.array(p['p_S_e1'])
        if p.get('p_S_e2') is not None:
            meas.p_S_e2 = np.array(p['p_S_e2'])
        if 'scan_pts_S' in p:
            meas.scan_pts_S = [np.array(pt) for pt in p['scan_pts_S']]

        # 关节角 (可选, 用于噪声注入)
        joints = None
        if 'J_i' in p and p['J_i'] is not None:
            joints = np.array(p['J_i'])

        records.append(CalibRecord(pose=pose, meas=meas, joints=joints))

    scene_gt = None
    if 'scene' in raw:
        scene_gt = SceneGT(
            R_he=np.array(raw['scene']['R_he_gt']),
            t_he=np.array(raw['scene']['t_he_gt']),
        )

    return CalibData(records=records, scene_gt=scene_gt)
