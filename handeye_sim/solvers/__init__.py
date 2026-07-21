#!/usr/bin/env python3
"""solvers/__init__.py — 统一求解器注册表"""

from handeye_sim.solvers.combined_9dof import calibrate_9dof
from handeye_sim.solvers.cross_12dof import calibrate_12dof_cross
from handeye_sim.solvers.iterative_he import calibrate_iterative
from handeye_sim.solvers.cross_12dof_v2 import calibrate_12dof_v2

# 求解器注册表
SOLVER_REGISTRY = {
    '9dof': (calibrate_9dof, '9-DOF plane+edge joint LM'),
    '12dof-cross': (calibrate_12dof_cross, '12-DOF C-anchored cross-product LM'),
    '12dof-v2': (calibrate_12dof_v2, '12-DOF v2: variable projection + scalar residuals'),
    'iterative': (calibrate_iterative, 'Alternating PCA→LM refinement'),
    'all': None,
}


def run_all_solvers(poses, meas, R_he_nom=None, t_he_nom=None,
                    solvers_cfg=None, method_to_cfg=None) -> dict:
    """运行所有求解器，返回 {method: CalibResult}"""
    if method_to_cfg is None:
        method_to_cfg = {}
    if solvers_cfg is None:
        solvers_cfg = {}
    results = {}
    for name in ['9dof', '12dof-cross', '12dof-v2', 'iterative']:
        fn, _ = SOLVER_REGISTRY[name]
        cfg = method_to_cfg.get(name, {})
        try:
            r = fn(poses, meas, R_he_nom, t_he_nom, solver_cfg=cfg)
            results[name] = r
        except Exception as e:
            results[name] = f"FAILED: {e}"
    return results


__all__ = [
    'calibrate_9dof', 'calibrate_12dof_cross', 'calibrate_iterative',
    'SOLVER_REGISTRY', 'run_all_solvers',
]
