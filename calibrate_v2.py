#!/usr/bin/env python3
"""
calibrate_v2.py — 手眼标定统一入口 v2

用法:
  python3 calibrate_v2.py solve -i data/xxx.json --method all
  python3 calibrate_v2.py solve -i data/xxx.json --method 12dof-cross
  python3 calibrate_v2.py collect --mode manual -o data/xxx.json
  python3 calibrate_v2.py full --mode auto --method all

所有可调参数在 config.yaml 中。
"""

import argparse, sys, os, yaml, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from handeye_sim.core.so3 import so3_exp, so3_log, rotation_error_deg, translation_error_mm
from handeye_sim.core.types import CalibResult
from handeye_sim.collection.io import save_calib_data, load_calib_data
from handeye_sim.solvers import SOLVER_REGISTRY, run_all_solvers


def load_config(path=None):
    """加载 config.yaml, 返回字典"""
    cfg_path = path or os.path.join(os.path.dirname(__file__), 'config.yaml')
    if not os.path.exists(cfg_path):
        print(f"[WARN] Config not found: {cfg_path}, using defaults")
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════
# 子命令: collect
# ═══════════════════════════════════════════════════════════════

def cmd_collect(args):
    cfg = load_config(args.config)
    coll_cfg = cfg.get('collection', {}).get('manual', {})
    output = args.output or cfg.get('data', {}).get('default_output', 'data/manual_test.json')

    if args.mode == 'manual':
        print(f"Manual collection → {output}")
        print("""
  In Docker container:
    python3 src/handeye_sim_bridge/handeye_sim_bridge/manual_recorder.py -o /workspace/data/xxx.json
  Keys: r=record  s=save  q=quit  p=preview
""")
    elif args.mode == 'auto':
        print("Auto collection: use auto_calib_v2_node in ROS2 container")
        print("  cd ros2_ws && ./start.sh")
        print("  ros2 run handeye_sim_bridge auto_calib_v2_node")
    return 0


# ═══════════════════════════════════════════════════════════════
# 子命令: solve
# ═══════════════════════════════════════════════════════════════

def cmd_solve(args):
    input_file = args.input
    if not os.path.exists(input_file):
        print(f"[ERROR] File not found: {input_file}"); return 1

    cfg = load_config(args.config)
    scene_cfg = cfg.get('scene', {})
    noise_cfg = scene_cfg.get('noise', {})
    perturb_cfg = scene_cfg.get('handeye_nominal_perturb', {})

    print(f"Loading data: {input_file}")
    data = load_calib_data(input_file)
    poses, meas = data.get_raw()
    print(f"  {data.n_poses()} poses  (e1={data.n_e1()}, e2={data.n_e2()})")

    # 名义手眼: 用 GT + 配置的扰动量
    rng = np.random.RandomState(42)
    R_pert_deg = perturb_cfg.get('R_perturb_deg', 5.0)
    t_pert_mm = perturb_cfg.get('t_perturb_mm', 12.0)

    if data.scene_gt is not None:
        w_pert = rng.randn(3) * np.deg2rad(R_pert_deg) / 3.0  # ~N(0, deg/3°)
        R_he_nom = data.scene_gt.R_he @ so3_exp(w_pert)
        t_he_nom = data.scene_gt.t_he + rng.randn(3) * t_pert_mm / 1000.0
        print(f"  Nominal HE: R_err={rotation_error_deg(R_he_nom, data.scene_gt.R_he):.2f}°  "
              f"(perturb: {R_pert_deg}°/{t_pert_mm}mm)")
    else:
        R_he_nom = np.eye(3); t_he_nom = np.zeros(3)

    # Tilt 统计
    tilts = [np.rad2deg(np.arccos(np.clip(abs(np.dot(R_i @ R_he_nom[:, 2], [0,0,1])), 0, 1)))
             for (R_i, _) in poses]
    print(f"  Tilt: min={min(tilts):.1f}° max={max(tilts):.1f}° mean={np.mean(tilts):.1f}°")

    # 运行求解器
    method = args.method
    print(f"\n{'='*60}")
    if method == 'all':
        results = run_all_solvers(poses, meas, R_he_nom, t_he_nom)
    else:
        fn, desc = SOLVER_REGISTRY.get(method, (None, None))
        if fn is None:
            print(f"Unknown method: {method}. Available: {list(SOLVER_REGISTRY.keys())}"); return 1
        print(f"Running {method}: {desc}")
        try:
            results = {method: fn(poses, meas, R_he_nom, t_he_nom)}
        except Exception as e:
            print(f"  FAILED: {e}"); import traceback; traceback.print_exc(); return 1

    print(f"\n{'='*60}\nResults:\n{'='*60}")
    for name, r in results.items():
        if isinstance(r, str): print(f"  [{name}] {r}")
        else: print(f"  {r.summary(data.scene_gt)}")

    # 达标检查
    diag_cfg = cfg.get('diagnostics', {})
    tilt_min = diag_cfg.get('tilt_min_deg', 10.0)
    if min(tilts) < tilt_min:
        print(f"\n  ⚠ Tilt min={min(tilts):.1f}° < {tilt_min}° — consider more diverse poses")
    return 0


# ═══════════════════════════════════════════════════════════════
# 子命令: full
# ═══════════════════════════════════════════════════════════════

def cmd_full(args):
    print("Full pipeline: collect → solve")
    ret = cmd_collect(args)
    if ret != 0: return ret
    solve_args = argparse.Namespace(
        input=args.output or 'data/recorded_poses.json',
        method=args.method, config=args.config,
    )
    return cmd_solve(solve_args)


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Hand-Eye Calibration CLI v2')
    sub = parser.add_subparsers(dest='command', required=True)

    # collect
    p = sub.add_parser('collect'); p.add_argument('--mode', choices=['auto','manual'], default='manual')
    p.add_argument('-o','--output'); p.add_argument('--n-poses', type=int, default=8)
    p.add_argument('-c','--config')

    # solve
    p = sub.add_parser('solve'); p.add_argument('-i','--input', required=True)
    p.add_argument('--method', default='all'); p.add_argument('-c','--config')

    # full
    p = sub.add_parser('full'); p.add_argument('--mode', default='auto')
    p.add_argument('--n-poses', type=int, default=8); p.add_argument('-o','--output')
    p.add_argument('--method', default='all'); p.add_argument('-c','--config')

    args = parser.parse_args()
    if args.command == 'collect': return cmd_collect(args)
    elif args.command == 'solve': return cmd_solve(args)
    elif args.command == 'full': return cmd_full(args)

if __name__ == '__main__':
    sys.exit(main())
