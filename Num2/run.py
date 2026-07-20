#!/usr/bin/env python3
"""
run.py — Num2 手眼标定统一入口

用法:
  python run.py                        # 使用 config.yaml 默认配置，单次标定
  python run.py --config my.yaml       # 使用自定义配置
  python run.py --mc 30                # Monte Carlo (覆盖配置文件)
  python run.py --method tilted_corner # 覆盖方法选择
  python run.py --animation            # 动态采集动画模式
  python run.py --compare              # 对比所有方法

配置: config.yaml
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration_framework import CalibrationFramework


def main():
    parser = argparse.ArgumentParser(
        description='Num2 手眼标定统一框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py                              # 默认配置, 单次标定
  python run.py --mc 30                      # 30 次 Monte Carlo
  python run.py --method corner_12dof        # 12-DOF 角点法
  python run.py --method plane_edge_9dof     # 9-DOF 平面+边缘
  python run.py --method tilted_corner       # 倾斜角点法 (默认)
  python run.py --method all --mc 10         # 对比所有方法各10次
  python run.py --animation                  # 动画采集模式
  python run.py --config my_config.yaml      # 自定义配置
        """)
    parser.add_argument('--config', default='/home/z/research_contact_handeye/verification/Num2/config.yaml',
                        help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--mc', type=int, default=None,
                        help='Monte Carlo 次数 (覆盖配置文件)')
    parser.add_argument('--method', default=None,
                        help='方法选择 (覆盖配置文件): corner_12dof | plane_edge_9dof | tilted_corner | all')
    parser.add_argument('--animation', action='store_true',
                        help='强制使用动画采集模式')
    parser.add_argument('--noise', type=float, default=None,
                        help='激光噪声 σ mm (覆盖配置文件)')
    parser.add_argument('--n-poses', type=int, default=None,
                        help='位姿数 (覆盖配置文件)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--compare', action='store_true',
                        help='快捷对比所有方法 (等价于 --method all --mc 10)')
    parser.add_argument('--quiet', action='store_true',
                        help='减少输出')

    args = parser.parse_args()

    # 快捷模式
    if args.compare:
        args.method = 'all'
        if args.mc is None:
            args.mc = 10

    # 加载框架
    fw = CalibrationFramework(args.config)

    # 命令行覆盖
    if args.method:
        fw.method = args.method
        # 同步更新采集模式
        default_mode = fw.METHOD_POSE_MODE.get(args.method, 'auto_grid')
        fw.config.setdefault('pose_collection', {})['mode'] = default_mode
    if args.animation:
        fw.config.setdefault('pose_collection', {})['mode'] = 'animation'
    if args.noise is not None:
        fw.config.setdefault('environment', {})['laser_noise'] = args.noise
    if args.n_poses is not None:
        fw.config.setdefault('pose_collection', {})['n_poses'] = args.n_poses
    if args.quiet:
        fw.verbose = False
        fw.config.setdefault('output', {})['verbose'] = False
        fw.config['output']['show_svd'] = False
        fw.config['output']['show_pose_stats'] = False

    # 运行
    if args.mc:
        fw.run_mc(n_trials=args.mc)
    else:
        fw.run_single(seed=args.seed)


if __name__ == '__main__':
    main()
