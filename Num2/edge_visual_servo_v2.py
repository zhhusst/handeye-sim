#!/usr/bin/env python3
"""
edge_visual_servo.py v2 — 实用版: 两段伺服 + 明确角点旋转

核心变更:
  1. 沿边推进不追到角点尖 (stop at u=0.12)
  2. 有明确的两阶段: EDGE1_COLLECT → CORNER_TURN → EDGE2_COLLECT
  3. 角点处: 显式后退+旋转90°, 再重新找边

几何事实:
  400mm平板, FOV宽度27cm(@50cm standoff)
  → 距角点12cm处是安全的最后一个位置(板面够宽)
  → 角点处需要分离的旋转操作
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reproduction_scene import (
    compute_fov_plate_scanline, make_transform, rodrigues,
    generate_hand_eye_gt
)
from corner_scene import generate_corner_plane, so3_exp, so3_log


# ============================================================================
# 1. 特征提取
# ============================================================================

@dataclass
class ProfileFeatures:
    has_intersection: bool = False
    n_pts: int = 0
    x_min: float = 0.0; x_max: float = 0.0
    x_center: float = 0.0; x_span: float = 0.0
    z_mean: float = 0.0; z_std: float = 0.0
    has_e1: bool = False; has_e2: bool = False
    e1_x: float = 0.0; e2_x: float = 0.0
    e1_z: float = 0.0; e2_z: float = 0.0
    edge_gap: Optional[float] = None
    n_endpoints: int = 0


def extract_features(sl: dict) -> ProfileFeatures:
    f = ProfileFeatures()
    if not sl.get('has_intersection', False):
        return f
    f.has_intersection = True
    
    pts = sl['scan_pts_S']
    f.n_pts = len(pts)
    if f.n_pts > 0:
        x, z = pts[:, 0], pts[:, 2]
        f.x_min, f.x_max = float(x.min()), float(x.max())
        f.x_center = float((x.min() + x.max()) / 2)
        f.x_span = float(x.max() - x.min())
        f.z_mean = float(z.mean())
        f.z_std = float(z.std())
    
    for et, pt in sl['endpoints_S']:
        if et == 'e1': f.has_e1 = True; f.e1_x = pt[0]; f.e1_z = pt[2]
        if et == 'e2': f.has_e2 = True; f.e2_x = pt[0]; f.e2_z = pt[2]
    
    f.n_endpoints = len(sl['endpoints_S'])
    if f.has_e1 and f.has_e2:
        f.edge_gap = abs(f.e1_x - f.e2_x)
    return f


def get_scanline(T_B_H, T_S_H, scene):
    T_B_S = T_B_H @ T_S_H
    return compute_fov_plate_scanline(
        T_B_S[:3,:3], T_B_S[:3,3],
        scene['C'], scene['n_B'], scene['u_B'], scene['v_B'],
        scene['w'], scene['h'])


# ============================================================================
# 2. 运动工具
# ============================================================================

def sensor_delta(T_B_H, T_S_H, delta_sensor):
    """apply 6-DOF delta in sensor frame"""
    dv = delta_sensor[:3]
    domega = delta_sensor[3:6]
    n = np.linalg.norm(domega)
    if n < 1e-12:
        Rd = np.eye(3)
    else:
        Rd = rodrigues(domega / n, n)
    Td = make_transform(Rd, dv)
    T_SH_inv = np.linalg.inv(T_S_H)
    return T_B_H @ (T_S_H @ Td @ T_SH_inv)


def base_translate(T_B_H, delta_base):
    """apply 3-DOF translation in base frame"""
    T_new = T_B_H.copy()
    T_new[:3, 3] += delta_base
    return T_new


# ============================================================================
# 3. 单边伺服 (P控制器)
# ============================================================================

def servo_step(f: ProfileFeatures, target_e1_x=0.005):
    """基于e1_x偏差的P控制
    
    Returns:
        sensor_delta: [dx, dy, dz, droll, dpitch, dyaw]
    """
    delta = np.zeros(6)
    if not f.has_intersection or f.n_pts < 1:
        delta[2] = -0.01  # back up
        return delta
    
    if f.has_e1:
        err = f.e1_x - target_e1_x
        delta[0] = -0.0003 * err  # lateral correction (sensor x)
    
    z_err = f.z_mean - 0.50
    delta[2] = -0.0008 * z_err  # standoff correction
    
    delta[:3] = np.clip(delta[:3], -0.008, 0.008)
    return delta


# ============================================================================
# 4. 主伺服循环
# ============================================================================

@dataclass
class ServoResult:
    records: List  # list of (T_B_H, sl, features, phase, step)
    phases: List[str]
    n_steps: int
    success: bool


def run_servo(scene, T_S_H, T_B_H_init, max_steps=200,
              forward_speed=0.005, edge1_min_records=15):
    """两段伺服: 边1采集 → 角点旋转 → 边2采集
    
    Phases:
      EDGE1:      沿边1推进, 采集到edge1_min_records后触发旋转
      CORNER_TURN: 停止前进, 后退1cm, 绕传感器Z轴旋转90°
      EDGE2_RECOVER: 旋转后找边2
      EDGE2:      沿边2推进, 采集
      DONE:       结束
    """
    T_B_H = T_B_H_init.copy()
    records = []
    phases_run = ['START']
    phase = 'EDGE1'
    
    u_B = scene['u_B']
    v_B = scene['v_B']
    n_B = scene['n_B']
    C = scene['C']
    
    turn_progress = 0  # 0~1 for corner turn
    TURN_STEPS = 25
    turn_started = False
    
    edge1_count = 0
    edge2_count = 0
    lost_streak = 0
    
    for step in range(max_steps):
        sl = get_scanline(T_B_H, T_S_H, scene)
        f = extract_features(sl)
        
        # ==== Phase Machine ====
        if phase == 'EDGE1':
            # 1) 伺服: 保持FOV在边上
            d_servo = servo_step(f)
            # 2) 推进: 沿边缘方向 (基坐标系)
            base_fwd = -forward_speed * (u_B / np.linalg.norm(u_B))
            # 3) Wiggle: 正弦姿态调制
            pitch = 0.20 * np.sin(2 * np.pi * step / 8)
            yaw = 0.12 * np.sin(2 * np.pi * step / 12 + 0.5)
            d_wiggle = np.array([0., 0., 0., 0., pitch, yaw])
            
            # 应用
            sensor_total = d_servo + d_wiggle
            sensor_total[:3] *= 0.6  # 削弱伺服平移, 让wiggle主导姿态
            sensor_total[3:6] = np.clip(sensor_total[3:6], -0.25, 0.25)
            T_B_H = base_translate(T_B_H, base_fwd)
            T_B_H = sensor_delta(T_B_H, T_S_H, sensor_total)
            
            # 记录
            if f.has_intersection and f.n_pts >= 3 and (f.has_e1 or f.has_e2):
                if not records or np.linalg.norm(T_B_H[:3,3] - records[-1][0][:3,3]) > 0.003:
                    records.append((T_B_H.copy(), sl, f, 'EDGE1', step))
                    edge1_count += 1
            
            # 触发角点旋转
            pos = T_B_H[:3, 3]
            dist_along_u = np.dot(pos - C, u_B / np.linalg.norm(u_B))
            if edge1_count >= edge1_min_records and dist_along_u < 0.15:
                phase = 'CORNER_TURN'
                turn_progress = 0
                phases_run.append('CORNER_TURN')
                
                # 同时: 增大standoff (避免转的时候丢平板)
                d_back = -0.015 * n_B
                T_B_H = base_translate(T_B_H, d_back)
        
        elif phase == 'CORNER_TURN':
            # 后退 + 绕传感器Z轴旋转90°
            t = turn_progress / TURN_STEPS
            
            # 后退在前50%, 恢复在后50%
            if t < 0.5:
                back_frac = t / 0.5
                b = -back_frac * 0.02 * n_B
            else:
                back_frac = (1 - t) / 0.5
                b = -back_frac * 0.02 * n_B
            
            # 旋转: 在中间80%完成90°
            rot_t = np.clip((t - 0.1) / 0.8, 0, 1)
            smooth = np.sin(np.pi * rot_t)  # 正弦加速减速
            step_rot = smooth * (np.pi / 2) / (TURN_STEPS * 0.8)
            d_turn = np.array([0., 0., 0., 0., 0., step_rot])
            
            T_B_H = base_translate(T_B_H, b)
            T_B_H = sensor_delta(T_B_H, T_S_H, d_turn)
            
            # 伺服修正: 如果还有交点, 保持
            if f.has_intersection and f.n_pts >= 2:
                d_servo = servo_step(f) * 0.3
                T_B_H = sensor_delta(T_B_H, T_S_H, d_servo)
            
            turn_progress += 1
            if turn_progress >= TURN_STEPS:
                phase = 'EDGE2_RECOVER'
                phases_run.append('EDGE2_RECOVER')
                # 重置搜索计数器
                lost_streak = 0
        
        elif phase == 'EDGE2_RECOVER':
            # 旋转后: 找边2
            # 沿传感器X方向扫过板面
            d_search = np.array([0.004, 0., 0., 0., 0., 0.02])  # 轻微yaw摆动
            T_B_H = sensor_delta(T_B_H, T_S_H, d_search)
            
            if f.has_intersection and f.n_pts >= 3 and (f.has_e1 or f.has_e2):
                # 找到板了!
                phase = 'EDGE2'
                phases_run.append('EDGE2')
                print("  [servo] Found edge 2! step=%d pts=%d" % (step, f.n_pts))
        
        elif phase == 'EDGE2':
            # 沿边2推进 (方向: -v_B, 用e2作为参照)
            d_servo = servo_step(f)  # 保持边
            base_fwd = -forward_speed * (v_B / np.linalg.norm(v_B))
            pitch = 0.20 * np.sin(2 * np.pi * step / 8 + 1.0)
            yaw = 0.12 * np.sin(2 * np.pi * step / 12 + 2.0)
            d_wiggle = np.array([0., 0., 0., 0., pitch, yaw])
            
            sensor_total = d_servo + d_wiggle
            sensor_total[:3] *= 0.6
            sensor_total[3:6] = np.clip(sensor_total[3:6], -0.25, 0.25)
            T_B_H = base_translate(T_B_H, base_fwd)
            T_B_H = sensor_delta(T_B_H, T_S_H, sensor_total)
            
            if f.has_intersection and f.n_pts >= 3 and (f.has_e1 or f.has_e2):
                if not records or np.linalg.norm(T_B_H[:3,3] - records[-1][0][:3,3]) > 0.003:
                    records.append((T_B_H.copy(), sl, f, 'EDGE2', step))
                    edge2_count += 1
            
            if edge2_count >= edge1_min_records:
                phase = 'DONE'
        
        elif phase == 'DONE':
            break
        
        # 交线丢失恢复
        if not f.has_intersection and phase not in ('CORNER_TURN', 'DONE'):
            lost_streak += 1
            if lost_streak > 6:
                # 后退和摆动找板
                d_recover = np.array([0., 0., -0.01, 0., 0.05 * np.sin(step), 0.])
                T_B_H = sensor_delta(T_B_H, T_S_H, d_recover)
        else:
            lost_streak = 0
    
    phases_run.append(phase)
    return ServoResult(
        records=records,
        phases=phases_run,
        n_steps=step + 1,
        success=phase == 'DONE' or (edge1_count >= 5 and edge2_count >= 5)
    )


# ============================================================================
# 5. 验证标定
# ============================================================================

def run_calibration(scene, T_S_H, T_B_H_init, forward_speed=0.005, 
                    edge1_min_records=15, max_steps=200):
    """完整伺服→标定流程"""
    from nbv_edge_plane import combined_solve_lm, combined_errors
    
    R_he_gt, t_he_gt = T_S_H[:3,:3], T_S_H[:3,3]
    
    # 伺服
    result = run_servo(scene, T_S_H, T_B_H_init, max_steps,
                       forward_speed, edge1_min_records)
    
    print("='=60")
    print("Servo: steps=%d records=%d" % (result.n_steps, len(result.records)))
    print("Phases: %s" % (' -> '.join(result.phases)))
    print("Success: %s" % result.success)
    
    e1r = sum(1 for _,_,_,p,_ in result.records if p == 'EDGE1')
    e2r = sum(1 for _,_,_,p,_ in result.records if p == 'EDGE2')
    print("Edge1 records=%d Edge2 records=%d" % (e1r, e2r))
    
    if len(result.records) < 4:
        print("Too few records!")
        return result, None
    
    # 构建标定数据
    poses, meas = [], []
    for T_B_H, sl, f, phase, step in result.records:
        R_i, t_i = T_B_H[:3,:3], T_B_H[:3,3]
        m = {
            'p_S_plane': sl['scan_pts_S'],
            'valid_e1': f.has_e1, 'valid_e2': f.has_e2,
            'p_S_e1': None, 'p_S_e2': None,
        }
        for et, pt in sl['endpoints_S']:
            if et == 'e1': m['p_S_e1'] = pt
            elif et == 'e2': m['p_S_e2'] = pt
        poses.append((R_i, t_i))
        meas.append(m)
    
    # 标定 (平面+边缘)
    theta = combined_solve_lm(np.zeros(9), poses, meas)
    R_he = so3_exp(theta[0:3])
    t_he = theta[3:6]
    
    R_err = np.rad2deg(np.linalg.norm(so3_log(R_he.T @ R_he_gt)))
    t_err = np.linalg.norm(t_he - t_he_gt) * 1000
    
    print("Calibration:")
    print("  R error: %.4f deg" % R_err)
    print("  t error: %.2f mm" % t_err)
    
    return result, (R_err, t_err)


# ============================================================================
# Demo
# ============================================================================

if __name__ == '__main__':
    seed = 42
    rng = np.random.default_rng(seed)
    C, n_B, u_B, v_B, d_1, d_2, w_m, h_m = generate_corner_plane(
        rng, plate_w=400, plate_h=500, alpha=np.pi/2)
    w_m, h_m = 0.4, 0.5
    
    scene = {'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B, 'w': w_m, 'h': h_m}
    T_S_H = generate_hand_eye_gt()
    
    print("=" * 60)
    print("Edge Visual Servo v2")
    print("=" * 60)
    print("Scene: C=[%.3f,%.3f,%.3f], n_B=[%.3f,%.3f,%.3f]" % (
        C[0],C[1],C[2], n_B[0],n_B[1],n_B[2]))
    print("Hand-eye: R=%s t=%s" % (np.round(so3_log(T_S_H[:3,:3]),4), 
                                     np.round(T_S_H[:3,3],4)))
    
    # 找有效初始位姿
    from nbv_edge_plane import _build_R_edge
    found = False
    for u_off in [0.30, 0.25]:
        for v_off in [0.03, 0.02]:
            for pitch in [-10, -15, -20]:
                for yaw in [0, 5]:
                    for so in [0.50, 0.55]:
                        target = C + u_off * u_B + v_off * v_B
                        R_S = _build_R_edge(pitch, yaw, v_B, n_B, u_B, v_B)
                        sp = target + so * n_B
                        sl = compute_fov_plate_scanline(
                            R_S, sp, C, n_B, u_B, v_B, w_m, h_m)
                        f = extract_features(sl)
                        if f.has_intersection and f.n_pts >= 3 and f.has_e1:
                            T_B_H_init = make_transform(R_S, sp) @ np.linalg.inv(T_S_H)
                            found = True
                            break
                    if found: break
                if found: break
            if found: break
        if found: break
    
    if not found:
        print("No valid initial pose!")
        sys.exit(1)
    
    print("Init: u=%.2f v=%.3f pitch=%d yaw=%d pts=%d e1_x=%.4f" % (
        u_off, v_off, pitch, yaw, f.n_pts, f.e1_x))
    
    # 跑伺服+标定
    result, calib = run_calibration(scene, T_S_H, T_B_H_init,
                                    forward_speed=0.005,
                                    edge1_min_records=10,
                                    max_steps=200)
    
    # 多次测试
    print("\n" + "=" * 60)
    print("Multiple runs (different seeds)")
    print("=" * 60)
    for seed in [42, 43, 44, 45, 46]:
        rng2 = np.random.default_rng(seed)
        C2, n_B2, u_B2, v_B2, _, _, w2, h2 = generate_corner_plane(
            rng2, plate_w=400, plate_h=500, alpha=np.pi/2)
        w2, h2 = 0.4, 0.5
        scene2 = {'C': C2, 'n_B': n_B2, 'u_B': u_B2, 'v_B': v_B2, 'w': w2, 'h': h2}
        T_S_H2 = generate_hand_eye_gt()
        
        # 不同的初始化
        found2 = False
        for u_off in [0.30, 0.25]:
            for v_off in [0.03, 0.02]:
                for pitch in [-10, -15]:
                    for yaw in [0, 5]:
                        target = C2 + u_off * u_B2 + v_off * v_B2
                        R_S = _build_R_edge(pitch, yaw, v_B2, n_B2, u_B2, v_B2)
                        sp = target + 0.50 * n_B2
                        sl = compute_fov_plate_scanline(
                            R_S, sp, C2, n_B2, u_B2, v_B2, w2, h2)
                        f = extract_features(sl)
                        if f.has_intersection and f.n_pts >= 3 and f.has_e1:
                            T_B_H_init2 = make_transform(R_S, sp) @ np.linalg.inv(T_S_H2)
                            found2 = True
                            break
                    if found2: break
                if found2: break
            if found2: break
        
        if not found2:
            print("Seed %d: no valid init" % seed)
            continue
        
        res2, cal2 = run_calibration(scene2, T_S_H2, T_B_H_init2,
                                      forward_speed=0.005,
                                      edge1_min_records=10,
                                      max_steps=200)
        if cal2 is not None:
            print("  -> R_err=%.4fdeg t_err=%.2fmm" % cal2)
        else:
            print("  -> calibration failed")
