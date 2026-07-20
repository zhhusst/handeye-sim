#!/usr/bin/env python3
"""Phase 0 closed-form solver based on Xu et al. (2022) straight-edge method.

Given 2+ rotation groups with 2+ translations each, solves R_he and t_he
via linear least squares (SVD), then projects to SO(3).

References:
  Xu et al., "Hand-eye calibration for 2D laser profile scanners using
  straight edges of common objects", RCIM 73 (2022) 102221.
"""

import numpy as np


def skew(a):
    """Skew-symmetric matrix [a]×."""
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])


def so3_exp(w):
    theta = np.linalg.norm(w)
    if theta < 1e-10:
        return np.eye(3)
    k = w / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def so3_log(R):
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = np.arccos(tr)
    if abs(theta) < 1e-10:
        return np.zeros(3)
    return theta / (2 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def project_so3(M):
    """Project 3x3 matrix to nearest SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(M)
    d = np.eye(3)
    d[2, 2] = np.linalg.det(U) * np.linalg.det(Vt)
    return U @ d @ Vt


def solve_rotation_xu(groups, verbose=False):
    """Closed-form R_he via Xu's method.

    Args:
        groups: list of rotation groups. Each group is a list of poses.
            Each pose = (R_BH, t_BH, e_pt) where
                R_BH: 3x3 hand rotation
                t_BH: 3x1 hand translation
                e_pt: 3x1 edge point in sensor frame (p_S_e1 or p_S_e2)

    Returns:
        R_he: 3x3 estimated hand-eye rotation
        success: bool
    """
    # For groups with < 2 poses, we can't form vectors
    valid_groups = [g for g in groups if len(g) >= 2]
    if len(valid_groups) < 2:
        if verbose:
            print(f"  Xu rotation: need ≥2 groups with ≥2 poses, got {len(valid_groups)}")
        return None, False

    Ar_blocks = []
    br_blocks = []

    for g_idx, group in enumerate(valid_groups):
        n = len(group)
        # Reference pose (first in group)
        Rb0, tb0, ps0 = group[0]

        for i in range(1, n):
            Rbi, tbi, psi = group[i]

            # us[i] = ps[i] - ps[0]  (sensor-frame vector)
            us_i = psi - ps0
            # ut[i] = tb[i] - tb[0]  (robot translation vector)
            ut_i = tbi - tb0

            for j in range(i + 1, n):
                Rbj, tbj, psj = group[j]
                us_j = psj - ps0
                ut_j = tbj - tb0

                # B1 = ((us[i] × us[j])^T ⊗ Rb[0])
                cross_us = np.cross(us_i, us_j)
                B1 = np.kron(cross_us.reshape(1, 3), Rb0)  # 3x9

                # B2 = (us[i]^T ⊗ Rb[0])
                B2 = np.kron(us_i.reshape(1, 3), Rb0)  # 3x9

                # B3 = (us[j]^T ⊗ Rb[0])
                B3 = np.kron(us_j.reshape(1, 3), Rb0)  # 3x9

                # br = -ut[i] × ut[j]
                br = -np.cross(ut_i, ut_j)  # 3x1

                # A1 = B1
                A1 = B1
                # A2 = [ut[j]]× · B2
                A2 = skew(ut_j) @ B2
                # A3 = [ut[i]]× · B3
                A3 = skew(ut_i) @ B3

                Ar_block = A1 + A2 - A3  # 3x9

                Ar_blocks.append(Ar_block)
                br_blocks.append(br)

    if len(Ar_blocks) == 0:
        return None, False

    Ar = np.vstack(Ar_blocks)
    br = np.hstack(br_blocks)

    # SVD solve: Ar · rs = br
    U, S, Vt = np.linalg.svd(Ar, full_matrices=False)
    # rs = V · S⁻¹ · U^T · br
    S_inv = np.diag(1.0 / np.maximum(S, 1e-12))
    rs = Vt.T @ S_inv @ U.T @ br

    # Reshape to 3x3 and project to SO(3)
    Rs = rs.reshape(3, 3, order='F')  # column-major since rs = vec(R)
    R_he = project_so3(Rs)

    if verbose:
        # Check residual
        residual = np.linalg.norm(Ar @ rs - br)
        print(f"  Xu rotation: {len(valid_groups)} groups, {len(Ar_blocks)} eq pairs, residual={residual:.6f}")

    return R_he, True


def solve_translation_xu(groups, R_he, verbose=False):
    """Closed-form t_he via Xu's method (step 2).

    Args:
        groups: same format as solve_rotation_xu
        R_he: estimated hand-eye rotation (from step 1)

    Returns:
        t_he: 3x1 estimated hand-eye translation
        success: bool
    """
    # Build all world-frame edge points
    all_points_w = []
    all_poses = []
    for group in groups:
        for Rb, tb, ps in group:
            pw = Rb @ R_he @ ps + tb + Rb @ np.zeros(3)  # t_he not yet known
            all_points_w.append(pw)
            all_poses.append((Rb, tb, ps))

    if len(all_points_w) < 2:
        return None, False

    # Estimate edge direction via PCA
    pts = np.array(all_points_w)
    centroid = np.mean(pts, axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    u_b = Vt[0]  # principal direction = edge direction

    At_blocks = []
    bt_blocks = []

    for i in range(len(all_poses)):
        for j in range(i + 1, len(all_poses)):
            Rbi, tbi, psi = all_poses[i]
            Rbj, tbj, psj = all_poses[j]

            # C = Rb[j] - Rb[i]
            C = Rbj - Rbi

            # bt = -(Rb[j] Rs ps[j] - Rb[i] Rs ps[i] + Tb[j] - Tb[i]) × u_b
            pwi = Rbi @ R_he @ psi
            pwj = Rbj @ R_he @ psj
            bt = -np.cross(pwj - pwi + tbj - tbi, u_b)

            # At = [u_b]× · C
            At = skew(u_b) @ C  # 3x3

            At_blocks.append(At)
            bt_blocks.append(bt)

    if len(At_blocks) == 0:
        return None, False

    At = np.vstack(At_blocks)
    bt = np.hstack(bt_blocks)

    U, S, Vt_svd = np.linalg.svd(At, full_matrices=False)
    S_inv = np.diag(1.0 / np.maximum(S, 1e-12))
    t_he = Vt_svd.T @ S_inv @ U.T @ bt

    if verbose:
        residual = np.linalg.norm(At @ t_he - bt)
        print(f"  Xu translation: {len(At_blocks)} pairs, residual={residual:.6f}")

    return t_he, True


def estimate_board_from_Rhe(records, R_he_est, verbose=False):
    """Estimate board parameters u_B, v_B, n_B from edge points.

    Uses the estimated R_he to transform e1/e2 from sensor to world frame,
    then fits lines via PCA for each edge.

    Args:
        records: list of dicts with 'R_BH', 't_BH', 'p_S_e1', 'p_S_e2', 'valid_e1', 'valid_e2'
        R_he_est: estimated hand-eye rotation

    Returns:
        (u_B, v_B, n_B) or (None, None, None) on failure
    """
    e1_world = []
    e2_world = []

    for rec in records:
        R_BH = rec['R_BH']
        t_BH = rec['t_BH']
        if rec.get('valid_e1') and 'p_S_e1' in rec:
            p_S = rec['p_S_e1']
            p_w = R_BH @ R_he_est @ p_S + t_BH
            e1_world.append(p_w)
        if rec.get('valid_e2') and 'p_S_e2' in rec:
            p_S = rec['p_S_e2']
            p_w = R_BH @ R_he_est @ p_S + t_BH
            e2_world.append(p_w)

    if len(e1_world) < 2 or len(e2_world) < 2:
        if verbose:
            print(f"  Board est: need ≥2 e1 and ≥2 e2 points, got {len(e1_world)}/{len(e2_world)}")
        return None, None, None

    e1_pts = np.array(e1_world)
    e2_pts = np.array(e2_world)

    # PCA for each edge
    c1 = np.mean(e1_pts, axis=0)
    _, _, Vt1 = np.linalg.svd(e1_pts - c1, full_matrices=False)
    d1 = Vt1[0]

    c2 = np.mean(e2_pts, axis=0)
    _, _, Vt2 = np.linalg.svd(e2_pts - c2, full_matrices=False)
    d2 = Vt2[0]

    # Determine which is u_B (closer to world X) and which is v_B
    if abs(np.dot(d1, [1, 0, 0])) > abs(np.dot(d2, [1, 0, 0])):
        u_B = d1 / np.linalg.norm(d1)
        v_B = d2 / np.linalg.norm(d2)
    else:
        u_B = d2 / np.linalg.norm(d2)
        v_B = d1 / np.linalg.norm(d1)

    n_B = np.cross(u_B, v_B)
    n_B /= np.linalg.norm(n_B)
    if n_B[2] < 0:
        n_B = -n_B

    if verbose:
        print(f"  Board est: u_B=[{u_B[0]:.4f},{u_B[1]:.4f},{u_B[2]:.4f}]")
        print(f"             v_B=[{v_B[0]:.4f},{v_B[1]:.4f},{v_B[2]:.4f}]")
        print(f"             n_B=[{n_B[0]:.4f},{n_B[1]:.4f},{n_B[2]:.4f}]")
        orth = abs(np.dot(u_B, v_B))
        print(f"             u⊥v={orth:.6f}")

    return u_B, v_B, n_B
