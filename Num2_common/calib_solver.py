"""
calib_solver.py — plane_edge_9dof 标定求解器

纯 numpy，无 ROS 依赖。可直接在容器内外使用。

依赖:
  - numpy
  - fov_geometry.py (在同一目录下) — 仅用于 so3_exp/so3_log

用法:
  from calib_solver import combined_solve_lm, CombinedResiduals
  
  theta_opt = combined_solve_lm(
      theta_init=np.zeros(9),
      poses=[(R_i, t_i), ...],  # robot hand poses
      measurements=[{'p_S_plane': ..., 'p_S_e1': ..., ...}, ...]
  )
"""

import numpy as np
from fov_geometry import so3_exp, so3_log


def combined_residuals(theta, poses, meas, w_plane=0.1, w_edge=1.0):
    """9-DOF: plane centered + edge collinearity
    
    Args:
        theta: [w_he(3), t_he(3), w_pl(3)]
        poses: [(R_i, t_i), ...] 机器人手部位姿
        meas: [{'p_S_plane': (N,3), 'p_S_e1': (3,) or None, 'p_S_e2': (3,) or None, 
                'valid_e1': bool, 'valid_e2': bool}, ...]
        w_plane, w_edge: 权重
    
    Returns:
        r: 残差向量
        mask: bool mask of valid residuals
        info: {'e1_pairs': n, 'e2_pairs': n, 'n_plane': n}
    """
    w_he, t_he, w_pl = theta[0:3], theta[3:6], theta[6:9]
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]
    
    plane_vals = []
    p_base_e1 = []
    p_base_e2 = []
    
    for (R_i, t_i), m in zip(poses, meas):
        R_BS = R_i @ R_he
        t_BS = t_i + R_i @ t_he
        
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            p_base_e1.append(R_BS @ m['p_S_e1'] + t_BS)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            p_base_e2.append(R_BS @ m['p_S_e2'] + t_BS)
        
        for p_S in m.get('p_S_plane', []):
            plane_vals.append(np.dot(n_B, R_BS @ p_S + t_BS))
    
    # Plane: global centered (2D line scanner scan lines are 1D per pose,
    # per-pose centering would eliminate all plane information)
    plane_vals = np.array(plane_vals)
    if len(plane_vals) > 0:
        plane_vals = plane_vals - np.mean(plane_vals)
    
    residuals = []
    mask = []
    wp = np.sqrt(w_plane)
    we = np.sqrt(w_edge)
    info = {'e1_pairs': 0, 'e2_pairs': 0, 'n_plane': len(plane_vals)}
    
    for v in plane_vals:
        residuals.append(v * wp)
        mask.append(True)
    
    # Edge 1 collinearity
    for k in range(len(p_base_e1) - 1):
        r = np.cross(p_base_e1[k+1] - p_base_e1[k], u_B)
        residuals.extend((r * we).tolist())
        mask.extend([True, True, True])
        info['e1_pairs'] += 1
    
    # Edge 2 collinearity
    for k in range(len(p_base_e2) - 1):
        r = np.cross(p_base_e2[k+1] - p_base_e2[k], v_B)
        residuals.extend((r * we).tolist())
        mask.extend([True, True, True])
        info['e2_pairs'] += 1
    
    return np.array(residuals), np.array(mask), info


def combined_cost(theta, poses, meas, w_plane=0.1, w_edge=1.0):
    r, mask, _ = combined_residuals(theta, poses, meas, w_plane, w_edge)
    return 0.5 * np.dot(r[mask], r[mask])


def combined_jacobian(theta, poses, meas, w_plane=0.1, w_edge=1.0, eps=1e-6):
    """数值 Jacobian"""
    r0, mask, info = combined_residuals(theta, poses, meas, w_plane, w_edge)
    n_params = len(theta)
    J = np.zeros((len(r0), n_params))
    
    for j in range(n_params):
        theta_plus = theta.copy()
        theta_plus[j] += eps
        r_plus, _, _ = combined_residuals(theta_plus, poses, meas, w_plane, w_edge)
        
        theta_minus = theta.copy()
        theta_minus[j] -= eps
        r_minus, _, _ = combined_residuals(theta_minus, poses, meas, w_plane, w_edge)
        
        J[:, j] = (r_plus - r_minus) / (2 * eps)
    
    return J, r0, mask, info


def combined_solve_lm(theta_init, poses, meas, w_plane=0.1, w_edge=1.0,
                      max_iter=200, tol=1e-12, lam0=1e-6):
    """LM 求解器"""
    theta = theta_init.copy()
    lam = lam0
    
    for it in range(max_iter):
        J, r, mask, _ = combined_jacobian(theta, poses, meas, w_plane, w_edge)
        rv = r[mask]
        Jv = J[mask, :]
        
        if len(rv) == 0:
            break
        
        cost = 0.5 * np.dot(rv, rv)
        H = Jv.T @ Jv
        g = Jv.T @ rv
        
        try:
            delta = -np.linalg.solve(H + lam * np.eye(9), g)
        except np.linalg.LinAlgError:
            lam *= 10
            continue
        
        tn = theta + delta
        new_cost = combined_cost(tn, poses, meas, w_plane, w_edge)
        
        if new_cost < cost:
            theta = tn
            lam = max(lam / 3, 1e-12)
            if abs(cost - new_cost) < tol:
                break
        else:
            lam = min(lam * 3, 1e6)
    
    return theta


def compute_errors(theta_est, theta_gt):
    """计算 R/t 误差
    Returns:
        R_err_deg, t_err_mm
    """
    Re = so3_exp(theta_est[0:3])
    Rg = so3_exp(theta_gt[0:3])
    Rd = Re.T @ Rg
    tr = np.clip((np.trace(Rd) - 1) / 2, -1, 1)
    R_err = np.rad2deg(np.arccos(tr))
    t_err = np.linalg.norm(theta_est[3:6] - theta_gt[3:6]) * 1000
    return R_err, t_err


def compute_information_matrix(poses, meas, theta, w_plane=0.1, w_edge=1.0, eps=1e-6):
    """计算 Fisher 信息矩阵 I = J^T Σ^{-1} J
    
    Args:
        poses, meas: 标定数据
        theta: 当前参数估计
        w_plane, w_edge: 权重
    
    Returns:
        I: 9×9 信息矩阵
    """
    J, r, mask, _ = combined_jacobian(theta, poses, meas, w_plane, w_edge, eps)
    Jv = J[mask, :]
    # 假设单位权重协方差
    I = Jv.T @ Jv
    return I


def compute_d_optimality(I_current, J_cand):
    """计算候选位姿的 D-optimality 信息增益"""
    I_new = I_current + J_cand.T @ J_cand
    return np.linalg.det(I_new) / np.linalg.det(I_current)


# ============================================================================
# 闭式解 (Xu 直边法)
# ============================================================================

def solve_he_closed_form(poses, meas, angle_threshold_deg=3.0, max_iter=10):
    """闭式解 R_he (Xu 直边法)"""
    v_obs = []
    for (R_i, t_i), m in zip(poses, meas):
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            v_obs.append((R_i, m['p_S_e1']))
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            v_obs.append((R_i, m['p_S_e2']))
    if len(v_obs) < 3: return None, None
    pairs = []
    for i in range(len(v_obs)):
        for j in range(i+1, len(v_obs)):
            Ri, pSi = v_obs[i]; Rj, pSj = v_obs[j]
            if np.rad2deg(np.linalg.norm(so3_log(Ri.T @ Rj))) > angle_threshold_deg:
                pairs.append((Ri, pSi, Rj, pSj))
    if not pairs: return np.eye(3), None
    return np.eye(3), pairs


def solve_he_t_cf(poses, meas, R_cf_in=None):
    """闭式解 t_he"""
    if R_cf_in is None: R_cf, _ = solve_he_closed_form(poses, meas)
    else: R_cf = R_cf_in
    return (R_cf, np.zeros(3)), None


# ============================================================================
# 12-DOF LM (cross-product 边约束)
# ============================================================================

def residuals_12dof(theta, poses, meas, w_plane=0.1, w_edge=1.0):
    """12-DOF 残差"""
    w_he, t_he, w_pl, C = theta[0:3], theta[3:6], theta[6:9], theta[9:12]
    R_he = so3_exp(w_he); R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:,0], R_pl[:,1], R_pl[:,2]
    pv, p1, p2 = [], [], []
    for (R_i, t_i), m in zip(poses, meas):
        R_BS = R_i @ R_he; t_BS = t_i + R_i @ t_he
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            p1.append(R_BS @ m['p_S_e1'] + t_BS)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            p2.append(R_BS @ m['p_S_e2'] + t_BS)
        for pS in m.get('p_S_plane', []):
            pv.append(np.dot(n_B, R_BS @ pS + t_BS - C))
    pv = np.array(pv)
    if len(pv) > 0: pv -= np.mean(pv)
    r, mk = [], []
    wp, we = np.sqrt(w_plane), np.sqrt(w_edge)
    for v in pv: r.append(v*wp); mk.append(True)
    for k in range(len(p1)-1):
        cr = np.cross(p1[k+1]-p1[k], u_B)
        r.extend((cr*we).tolist()); mk.extend([True,True,True])
    for k in range(len(p2)-1):
        cr = np.cross(p2[k+1]-p2[k], v_B)
        r.extend((cr*we).tolist()); mk.extend([True,True,True])
    return np.array(r), np.array(mk)


def solve_12dof_lm(theta_init, poses, meas, max_iter=500, tol=1e-12):
    """12-DOF LM"""
    theta = theta_init.copy(); lam = 1e-4
    for _ in range(max_iter):
        r, mask = residuals_12dof(theta, poses, meas)
        rv = r[mask]; cost = 0.5*np.dot(rv, rv)
        eps = 1e-6; J = np.zeros((len(r), 12))
        for k in range(12):
            sp = theta.copy(); sp[k] += eps
            sm = theta.copy(); sm[k] -= eps
            J[:, k] = (residuals_12dof(sp,poses,meas)[0] - residuals_12dof(sm,poses,meas)[0])/(2*eps)
        Jv = J[mask, :]
        try: delta = -np.linalg.solve(Jv.T@Jv + lam*np.eye(12), Jv.T@rv)
        except np.linalg.LinAlgError: lam *= 10; continue
        tn = theta + delta
        cn = 0.5*np.dot(residuals_12dof(tn,poses,meas)[0][mask], residuals_12dof(tn,poses,meas)[0][mask])
        if cn < cost: theta = tn; lam = max(lam/3,1e-12)
        else: lam = min(lam*3,1e6)
        if abs(cost-cn) < tol: break
    return theta


def init_12dof(poses, meas, R_cf=None):
    """初始化 12-DOF"""
    if R_cf is None: R_cf, _ = solve_he_closed_form(poses, meas)
    if R_cf is None: R_cf = np.eye(3)
    (_, t_cf), _ = solve_he_t_cf(poses, meas, R_cf_in=R_cf)
    if np.linalg.norm(t_cf) < 1e-6: t_cf = np.zeros(3)
    p1, p2 = [], []
    for (R_i, t_i), m in zip(poses, meas):
        R_BS = R_i @ R_cf; t_BS = t_i + R_i @ t_cf
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            p1.append(R_BS @ m['p_S_e1'] + t_BS)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            p2.append(R_BS @ m['p_S_e2'] + t_BS)
    def fd(pts):
        if len(pts)<2: return np.array([1.,0.,0.])
        return np.linalg.svd(np.array(pts)-np.mean(pts,axis=0))[2][0]
    u_B=fd(p1); v_B=fd(p2)
    n_B=np.cross(u_B,v_B)
    if np.linalg.norm(n_B)>1e-6: n_B/=np.linalg.norm(n_B)
    else: n_B=np.array([0.,0.,1.])
    v_B=np.cross(n_B,u_B); v_B/=np.linalg.norm(v_B)
    R_pl=np.column_stack([u_B,v_B,n_B])
    p1r=np.mean(p1,axis=0) if p1 else np.zeros(3)
    p2r=np.mean(p2,axis=0) if p2 else np.zeros(3)
    sk=lambda v:np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    Ac=np.vstack([sk(u_B),sk(v_B),n_B.reshape(1,3)])
    bc=np.hstack([sk(u_B)@p1r,sk(v_B)@p2r,[np.dot(n_B,(p1r+p2r)/2)]])
    C,*_=np.linalg.lstsq(Ac,bc,rcond=None)
    return np.concatenate([so3_log(R_cf), t_cf, so3_log(R_pl), C])


def solve_12dof_with_restarts(poses, meas, n_restarts=20, seed=42, verbose=False,
                              w_he_init=None):
    """12-DOF 多重重启

    Args:
        w_he_init: 可选 (3,) 旋转向量，作为第一个重启的初始 R_he
    """
    rng = np.random.RandomState(seed)
    best_cost, best_theta = float('inf'), None
    n_good = 0

    # Xu 初始化 (如果提供)
    restarts_done = 0
    if w_he_init is not None:
        theta_init = init_12dof(poses, meas)
        theta_init[0:3] = w_he_init
        theta_opt = solve_12dof_lm(theta_init, poses, meas)
        r, mask = residuals_12dof(theta_opt, poses, meas)
        cost = 0.5*np.dot(r[mask], r[mask])
        if cost < best_cost: best_cost = cost; best_theta = theta_opt
        if cost < 1e-4: n_good += 1
        if verbose: print(f"  Xu-init: cost={cost:.2e}")
        restarts_done = 1

    for trial in range(n_restarts):
        if trial == 0: theta_init = init_12dof(poses, meas)
        else:
            ax = rng.randn(3); ax /= np.linalg.norm(ax)
            theta_init = init_12dof(poses, meas)
            theta_init[0:3] = ax * rng.uniform(0, np.pi)
        theta_opt = solve_12dof_lm(theta_init, poses, meas)
        r, mask = residuals_12dof(theta_opt, poses, meas)
        cost = 0.5*np.dot(r[mask], r[mask])
        if cost < best_cost: best_cost = cost; best_theta = theta_opt
        if cost < 1e-4: n_good += 1
        if verbose and (trial + restarts_done) % 5 == 0:
            print(f"  restart {trial}: cost={cost:.2e}")
    return best_theta, {'n_tried': n_restarts + restarts_done, 'n_good': n_good,
                         'best_cost': best_cost}


# ============================================================================
# PRINCIPLE.md 标量边约束
# ============================================================================

def residuals_principle(theta, poses, meas):
    """标量残差: v_B.(p-C)=0, n_perp2.(p-C)=0, n_B.(p-C)=0"""
    w_he, t_he = theta[0:3], theta[3:6]
    w_pl, C = theta[6:9], theta[9:12]
    R_he = so3_exp(w_he); R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:,0], R_pl[:,1], R_pl[:,2]
    n_perp2 = -u_B
    residuals = []
    for (R_i, t_i), m in zip(poses, meas):
        R_BS = R_i @ R_he; t_BS = t_i + R_i @ t_he
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            residuals.append(float(v_B @ (R_BS @ m['p_S_e1'] + t_BS - C)))
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            residuals.append(float(n_perp2 @ (R_BS @ m['p_S_e2'] + t_BS - C)))
        for pS in m.get('p_S_plane', []):
            residuals.append(float(n_B @ (R_BS @ pS + t_BS - C)))
    return np.array(residuals)


def solve_principle_lm(theta_init, poses, meas, max_iter=500, tol=1e-12):
    """LM 数值 Jacobian"""
    theta = theta_init.copy(); lam = 1e-4
    for _ in range(max_iter):
        r = residuals_principle(theta, poses, meas)
        cost = 0.5*np.dot(r, r)
        eps = 1e-6; J = np.zeros((len(r), 12))
        for j in range(12):
            tp = theta.copy(); tp[j] += eps
            tm = theta.copy(); tm[j] -= eps
            J[:,j] = (residuals_principle(tp,poses,meas) - residuals_principle(tm,poses,meas))/(2*eps)
        try: delta = -np.linalg.solve(J.T@J + lam*np.eye(12), J.T@r)
        except np.linalg.LinAlgError: lam *= 10; continue
        tn = theta + delta
        cn = 0.5*np.dot(residuals_principle(tn,poses,meas), residuals_principle(tn,poses,meas))
        if cn < cost: theta = tn; lam = max(lam/3, 1e-12)
        else: lam = min(lam*3, 1e6)
        if abs(cost-cn) < tol: break
    return theta


def solve_principle_with_restarts(poses, meas, n_restarts=20, seed=42,
                                   verbose=False, w_he_init=None):
    """多重重启

    Args:
        w_he_init: 可选 (3,) 旋转向量，作为第一个重启的初始 R_he
    """
    rng = np.random.RandomState(seed)
    best_cost, best_theta = float('inf'), None
    restarts_done = 0
    if w_he_init is not None:
        theta_init = np.zeros(12); theta_init[0:3] = w_he_init
        theta_opt = solve_principle_lm(theta_init, poses, meas, max_iter=300)
        cost = 0.5*np.dot(residuals_principle(theta_opt,poses,meas), residuals_principle(theta_opt,poses,meas))
        if cost < best_cost: best_cost = cost; best_theta = theta_opt
        if verbose: print(f"  Xu-init: cost={cost:.2e}")
        restarts_done = 1
    for trial in range(n_restarts):
        if trial == 0: theta_init = np.zeros(12)
        else:
            ax = rng.randn(3); ax /= np.linalg.norm(ax)
            theta_init = np.zeros(12)
            theta_init[0:3] = ax * rng.uniform(0, np.pi)
        theta_opt = solve_principle_lm(theta_init, poses, meas, max_iter=300)
        cost = 0.5*np.dot(residuals_principle(theta_opt,poses,meas), residuals_principle(theta_opt,poses,meas))
        if cost < best_cost: best_cost = cost; best_theta = theta_opt
        if verbose and (trial + restarts_done) % 5 == 0:
            print(f"  M2 restart {trial}: cost={cost:.3e}")
    return best_theta, {'n_tried': n_restarts, 'best_cost': best_cost}


# ============================================================================
# 方法3: PRINCIPLE.md 传感器帧预测 + gauge固定 (Num2 corner_calib.py)
# ============================================================================

def residuals_principle_12dof(theta, poses, meas, alpha=np.pi/2):
    """传感器帧预测残差 (PRINCIPLE.md Sec 4.3 + corner_calib.py)"""
    w_he, t_he, w_pl, C = theta[0:3], theta[3:6], theta[6:9], theta[9:12]
    R_he = so3_exp(w_he); R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:,0], R_pl[:,1], R_pl[:,2]
    d_1 = u_B; d_2 = np.cos(alpha)*u_B + np.sin(alpha)*v_B
    residuals, mask = [], []
    for (R_i, t_i), m in zip(poses, meas):
        R_BS = R_i @ R_he; t_BS = t_i + R_i @ t_he
        nl = R_BS[:, 1]
        # edge1 (2D sensor frame)
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            den = np.dot(nl, d_1)
            if abs(den) > 1e-12:
                s = np.dot(nl, t_BS-C)/den
                pS = R_he.T @ (R_i.T @ (C+s*d_1 - t_i) - t_he)
                residuals.extend([pS[0]-m['p_S_e1'][0], pS[2]-m['p_S_e1'][2]])
                mask.extend([True, True])
            else: residuals.extend([0.,0.]); mask.extend([False,False])
        else: residuals.extend([0.,0.]); mask.extend([False,False])
        # edge2 (2D)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            den = np.dot(nl, d_2)
            if abs(den) > 1e-12:
                s = np.dot(nl, t_BS-C)/den
                pS = R_he.T @ (R_i.T @ (C+s*d_2 - t_i) - t_he)
                residuals.extend([pS[0]-m['p_S_e2'][0], pS[2]-m['p_S_e2'][2]])
                mask.extend([True, True])
            else: residuals.extend([0.,0.]); mask.extend([False,False])
        else: residuals.extend([0.,0.]); mask.extend([False,False])
        # plane (1D)
        for pS in m.get('p_S_plane', []):
            residuals.append(float(n_B @ (R_BS @ pS + t_BS - C)))
            mask.append(True)
    return np.array(residuals), np.array(mask)


def cost_principle_12dof(theta, poses, meas, alpha=np.pi/2):
    r, mask = residuals_principle_12dof(theta, poses, meas, alpha)
    rv = r[mask]
    return 0.5*np.dot(rv, rv) if len(rv) > 0 else 1e30


def init_principle_12dof(poses, meas, R_he_nom, t_he_nom):
    """初始化 (名义手眼 + PCA)"""
    all_pts = []
    for (R_i, t_i), m in zip(poses, meas):
        R_BS = R_i @ R_he_nom; t_BS = t_i + R_i @ t_he_nom
        for pS in m.get('p_S_plane', []):
            all_pts.append(R_BS @ pS + t_BS)
    if not all_pts: return None, None
    all_pts = np.array(all_pts)
    c = np.mean(all_pts, axis=0)
    _, ev = np.linalg.eigh((all_pts-c).T @ (all_pts-c) / len(all_pts))
    n_B = ev[:,0]; n_B /= np.linalg.norm(n_B)
    ep = []
    for (R_i, t_i), m in zip(poses, meas):
        R_BS = R_i @ R_he_nom; t_BS = t_i + R_i @ t_he_nom
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            ep.append(R_BS @ m['p_S_e1'] + t_BS)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            ep.append(R_BS @ m['p_S_e2'] + t_BS)
    u_B = None
    if len(ep) >= 3:
        ec = np.array(ep) - np.mean(ep, axis=0)
        _, _, Vt = np.linalg.svd(ec, full_matrices=False)
        d1 = Vt[0] - np.dot(Vt[0], n_B)*n_B
        if np.linalg.norm(d1) > 1e-6:
            u_B = d1 / np.linalg.norm(d1)
    if u_B is None:
        u_B = np.cross(np.array([0.,0.,1.]), n_B) if abs(n_B[2])<0.9 else np.array([1.,0.,0.])
        u_B /= np.linalg.norm(u_B)
    v_B = np.cross(n_B, u_B); v_B /= np.linalg.norm(v_B)
    R_pl = np.column_stack([u_B, v_B, n_B])
    return np.concatenate([so3_log(R_he_nom), t_he_nom, so3_log(R_pl), c - np.dot(n_B,c)*n_B]), float(np.dot(n_B, c))


def solve_principle_12dof_lm(theta_init, poses, meas, alpha=np.pi/2,
                              max_iter=50, fix_C_proj=None):
    """LM + gauge固定"""
    theta = theta_init.copy(); lam = 1e-4
    for _ in range(max_iter):
        r, mask = residuals_principle_12dof(theta, poses, meas, alpha)
        rv = r[mask]
        if len(rv) == 0: break
        cost = 0.5*np.dot(rv, rv)
        eps = 1e-6; J = np.zeros((len(r), 12))
        for k in range(12):
            sp = theta.copy(); sp[k] += eps
            sm = theta.copy(); sm[k] -= eps
            rp, _ = residuals_principle_12dof(sp, poses, meas, alpha)
            rm, _ = residuals_principle_12dof(sm, poses, meas, alpha)
            J[:,k] = (rp-rm)/(2*eps)
        Jv = J[mask, :]
        try: delta = -np.linalg.solve(Jv.T@Jv + lam*np.eye(12), Jv.T@rv)
        except np.linalg.LinAlgError: lam *= 10; continue
        tn = theta + delta
        if fix_C_proj is not None:
            nB = so3_exp(tn[6:9])[:,2]
            tn[9:12] -= (np.dot(nB, tn[9:12]) - fix_C_proj)*nB
        cn = cost_principle_12dof(tn, poses, meas, alpha)
        if cn < cost: theta = tn; lam = max(lam/3, 1e-12)
        else: lam = min(lam*3, 1e6)
        if abs(cost-cn) < 1e-8: break
    return theta


def solve_principle_12dof_with_restarts(poses, meas, R_he_nom, t_he_nom,
                                         n_restarts=20, alpha=np.pi/2,
                                         seed=42, verbose=False,
                                         w_he_init=None):
    """PRINCIPLE.md 12-DOF + 多重重启

    Args:
        w_he_init: 可选 (3,) 旋转向量，作为第一个重启的初始 R_he
    """
    theta_nom, fix_C_proj = init_principle_12dof(poses, meas, R_he_nom, t_he_nom)
    if theta_nom is None: return None, {'error': 'init failed'}
    rng = np.random.RandomState(seed)
    best_cost, best_theta = float('inf'), None
    n_good = 0
    restarts_done = 0
    if w_he_init is not None:
        ti = theta_nom.copy(); ti[0:3] = w_he_init
        to = solve_principle_12dof_lm(ti, poses, meas, alpha, max_iter=50, fix_C_proj=fix_C_proj)
        cost = cost_principle_12dof(to, poses, meas, alpha)
        if cost < best_cost: best_cost = cost; best_theta = to
        if cost < 1e-3: n_good += 1
        if verbose: print(f"  Xu-init: cost={cost:.2e}")
        restarts_done = 1
    for trial in range(n_restarts):
        if trial == 0: theta_init = theta_nom.copy()
        else:
            ax = rng.randn(3); ax /= np.linalg.norm(ax)
            theta_init = theta_nom.copy()
            theta_init[0:3] += ax * rng.uniform(0, 0.5)
        theta_opt = solve_principle_12dof_lm(
            theta_init, poses, meas, alpha, max_iter=50, fix_C_proj=fix_C_proj)
        cost = cost_principle_12dof(theta_opt, poses, meas, alpha)
        if cost < best_cost: best_cost = cost; best_theta = theta_opt
        if cost < 1e-3: n_good += 1
        if verbose and (trial + restarts_done) % 5 == 0:
            print(f"  PRINCIPLE restart {trial}: cost={cost:.2e}")
    return best_theta, {'n_tried': n_restarts, 'n_good': n_good,
                         'best_cost': best_cost, 'fix_C_proj': fix_C_proj}


# ============================================================================
# Xu 2022: 同旋转数据闭式解 Rs
# ============================================================================

def solve_rs_xu2022(poses, meas, angle_tol_deg=1.0):
    """从同旋转、不同平移的位姿中估计 R_he (Xu 2022 式)

    原理: 相同法兰旋转 R_BH 下，边断点的差分消去 t_he，
    留下纯 R_he 约束。利用两正交边的平行性约束解 R_he。

    Args:
        poses: [(R_i, t_i), ...]
        meas: [dict with p_S_e1, p_S_e2, ...]
        angle_tol_deg: R_BH 差异阈值 (判定"同旋转")

    Returns:
        R_he: 3×3 估计，失败返回 None
    """
    if len(poses) < 3:
        return None

    # 找最大同旋转组
    groups = {}
    R0 = poses[0][0]
    for i, (R_i, _) in enumerate(poses):
        angle = np.rad2deg(np.linalg.norm(so3_log(R0.T @ R_i)))
        if angle < angle_tol_deg:
            group_id = 0
        else:
            group_id = i  # 不同旋转 → 新组
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(i)

    # 选最大的同旋转组
    best_gid = max(groups, key=lambda g: len(groups[g]))
    indices = groups[best_gid]
    if len(indices) < 3:
        return None

    # 收集该组的边断点 (传感器帧) 和机器人平移
    e1_pts, e2_pts, ts = [], [], []
    for idx in indices:
        R_i, t_i = poses[idx]
        m = meas[idx]
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            e1_pts.append(np.asarray(m['p_S_e1']))
            ts.append(t_i)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            e2_pts.append(np.asarray(m['p_S_e2']))
            # ts already appended

    if len(e1_pts) < 3 or len(e2_pts) < 3:
        return None

    # 估计边方向 (传感器帧): 差分的主方向
    def edge_direction(pts):
        if len(pts) < 2: return None
        diffs = np.array([pts[i] - pts[0] for i in range(1, len(pts))])
        # SVD 取主方向
        _, _, Vt = np.linalg.svd(diffs, full_matrices=False)
        d = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
        return d

    d1_S = edge_direction(e1_pts)
    d2_S = edge_direction(e2_pts)
    if d1_S is None or d2_S is None:
        return None

    # 约束: d1_B 和 d2_B 在基座标系应正交
    # d1_B = R_BH @ R_he @ d1_S, d2_B = R_BH @ R_he @ d2_S
    # d1_B · d2_B = 0 → d1_S^T @ R_he^T @ R_BH^T @ R_BH @ R_he @ d2_S = 0
    #              = d1_S^T @ R_he^T @ R_he @ d2_S = d1_S^T @ d2_S = 0 (恒等!)
    # 不对——R_BH^T @ R_BH = I，所以 d1_S · d2_S = 0 是传感器帧的正交性，
    # 不是基座标系的约束。

    # 正确约束: 利用机器人平移
    # ut[i] = t_i - t_0 (机器人平移)
    # us[i] = p_S[i] - p_S[0] (传感器帧边点差分)
    # ub[i] = R_BH @ R_he @ us[i] + ut[i]
    # 所有 ub[i] 平行 (同一边)

    # 简化: us[i] ≈ α_i * d_S (沿边方向)
    # ub[i] ≈ α_i * R_BH @ R_he @ d_S + ut[i]
    # 平行约束: ‖ub[i] × ub[j]‖ ≈ 0

    R_BH = poses[indices[0]][0]
    us_e1 = e1_pts[1] - e1_pts[0]
    us_e2 = e2_pts[1] - e2_pts[0]
    ut_e1 = ts[1] - ts[0]
    ut_e2 = ts[1] - ts[0]

    if np.linalg.norm(us_e1) < 1e-8 or np.linalg.norm(us_e2) < 1e-8:
        return None

    # 数值优化: 最小化 ‖ub_e1[1] × ub_e1[2]‖² + ‖ub_e2[1] × ub_e2[2]‖²
    # + 正交约束 ‖(R_BH @ R_he @ d1_S) · (R_BH @ R_he @ d2_S)‖²
    from scipy.optimize import minimize

    us_all = []
    ut_all = []
    for k in range(1, min(len(e1_pts), 3)):
        us_all.append(e1_pts[k] - e1_pts[0])
        ut_all.append(ts[k] - ts[0])
    for k in range(1, min(len(e2_pts), 3)):
        us_all.append(e2_pts[k] - e2_pts[0])
        ut_all.append(ts[k] - ts[0])

    def cost_rs(w):
        R = so3_exp(w)
        cost = 0.0
        # 平行约束
        for i in range(len(us_all)):
            ubi = R_BH @ R @ us_all[i] + ut_all[i]
            for j in range(i + 1, len(us_all)):
                ubj = R_BH @ R @ us_all[j] + ut_all[j]
                cost += np.sum(np.cross(ubi, ubj) ** 2)
        # 正交约束
        d1_B = R_BH @ R @ d1_S
        d2_B = R_BH @ R @ d2_S
        cost += 100.0 * np.dot(d1_B, d2_B) ** 2
        return cost

    # 从名义值出发
    w0 = np.zeros(3)
    res = minimize(cost_rs, w0, method='Nelder-Mead',
                   options={'maxiter': 500, 'xatol': 1e-8})
    R_est = so3_exp(res.x)
    return R_est


def iterative_refine_he(poses, meas, R_he_init, t_he_init, max_iter=5, verbose=False):
    """Sharifzadeh 2020 式迭代 refine: a) 投影点→拟合 n_B, b) n_B 约束→解手眼

    核心: 当前手眼投影激光点到基座标系 → PCA 拟合板法向 n_B →
    用 n_B 作为平面约束 → LM 求解手眼 → 重复。

    Args:
        poses: [(R_i, t_i), ...]
        meas: [dict with p_S_plane, p_S_e1, p_S_e2, ...]
        R_he_init, t_he_init: 初始手眼 (3×3, 3×1)
        max_iter: 最大迭代次数

    Returns:
        (R_he, t_he, R_pl, n_B, n_iter)
    """
    R_he = R_he_init.copy()
    t_he = t_he_init.copy()
    n_B, u_B, v_B = None, None, None

    for it in range(max_iter):
        # Step a: 投影所有板面激光点到基座标系 → PCA 拟合平面
        all_pts = []
        for (R_i, t_i), m in zip(poses, meas):
            R_BS = R_i @ R_he
            t_BS = t_i + R_i @ t_he
            for p_S in m.get('p_S_plane', []):
                all_pts.append(R_BS @ p_S + t_BS)

        if len(all_pts) < 10:
            break
        all_pts = np.array(all_pts)
        c = all_pts.mean(axis=0)
        _, ev = np.linalg.eigh(
            (all_pts - c).T @ (all_pts - c) / len(all_pts))
        n_B_new = ev[:, 0]
        n_B_new /= np.linalg.norm(n_B_new)

        # 方向一致性: n_B 指向传感器侧 (dot(n_B, z_S_avg) < 0)
        z_S_samples = []
        for R_i, _ in poses:
            z_S_samples.append(R_i @ R_he[:, 2])
        z_S_avg = np.mean(z_S_samples, axis=0)
        if np.dot(n_B_new, z_S_avg) > 0:
            n_B_new = -n_B_new

        # Step b: 用 n_B 约束 + 边约束解手眼 (9-DOF LM)
        u_B_new = (np.array([1., 0., 0.]) if abs(n_B_new[2]) < 0.9
                   else np.array([0., 1., 0.]))
        u_B_new -= np.dot(u_B_new, n_B_new) * n_B_new
        u_B_new /= np.linalg.norm(u_B_new) + 1e-12
        v_B_new = np.cross(n_B_new, u_B_new)
        v_B_new /= np.linalg.norm(v_B_new) + 1e-12
        R_pl_new = np.column_stack([u_B_new, v_B_new, n_B_new])

        theta_init = np.zeros(9)
        theta_init[0:3] = so3_log(R_he)
        theta_init[3:6] = t_he
        theta_init[6:9] = so3_log(R_pl_new)

        theta_opt = combined_solve_lm(
            theta_init, poses, meas,
            w_plane=0.1, w_edge=1.0, max_iter=100)

        R_he_new = so3_exp(theta_opt[0:3])
        t_he_new = theta_opt[3:6]

        # 收敛检查
        dR = np.linalg.norm(so3_log(R_he_new.T @ R_he))
        dt = np.linalg.norm(t_he_new - t_he)
        if verbose:
            print(f"  iter {it}: |dR|={np.rad2deg(dR):.2f}° |dt|={dt*1000:.2f}mm")
        if dR < 1e-4 and dt < 1e-6:
            R_he, t_he = R_he_new.copy(), t_he_new.copy()
            n_B, u_B, v_B = n_B_new, u_B_new, v_B_new
            return R_he, t_he, R_pl_new, n_B, it + 1

        R_he, t_he = R_he_new.copy(), t_he_new.copy()
        n_B, u_B, v_B = n_B_new, u_B_new, v_B_new

    return R_he, t_he, R_pl_new, n_B, max_iter
