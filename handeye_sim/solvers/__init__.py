#!/usr/bin/env python3
"""solvers/__init__.py — 统一求解器注册表"""

from handeye_sim.solvers.combined_9dof import calibrate_9dof
from handeye_sim.solvers.cross_12dof import calibrate_12dof_cross
from handeye_sim.solvers.iterative_he import calibrate_iterative

# 求解器注册表
SOLVER_REGISTRY = {
    '9dof': (calibrate_9dof, '9-DOF plane+edge joint LM'),
    '12dof-cross': (calibrate_12dof_cross, '12-DOF C-anchored cross-product LM'),
    'iterative': (calibrate_iterative, 'Alternating PCA→LM refinement'),
    'all': None,
}


def run_all_solvers(poses, meas, R_he_nom=None, t_he_nom=None) -> dict:
    """运行所有求解器，返回 {method: CalibResult}"""
    results = {}
    for name in ['9dof', '12dof-cross', 'iterative']:
        fn, _ = SOLVER_REGISTRY[name]
        try:
            r = fn(poses, meas, R_he_nom, t_he_nom)
            results[name] = r
        except Exception as e:
            results[name] = f"FAILED: {e}"
    return results


__all__ = [
    'calibrate_9dof', 'calibrate_12dof_cross', 'calibrate_iterative',
    'SOLVER_REGISTRY', 'run_all_solvers',
]
