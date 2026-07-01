#!/usr/bin/env python3
"""replay_calib_poses.py — 回放已保存的标定采集位姿

用法:
  ros2 run handeye_sim_bridge replay_calib_poses.py <记录文件.json>

流程:
  1. 加载 JSON 记录文件（含关节角）
  2. 对每个位姿，用 MoveIt2 规划并移动到保存的关节角
  3. 到达后等待用户确认:
     'r' — 记录当前位姿 (+ 采集 /gocator/profile)
     'n' — 跳过
     'q' — 退出回放
"""

import rclpy
from rclpy.node import Node
import numpy as np
import sys, os, json, time
import select

sys.path.insert(0, '/workspace/common')
from fov_geometry import so3_log


def main():
    if len(sys.argv) < 2:
        print("用法: python3 replay_calib_poses.py <记录文件.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    with open(json_path) as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {json_path}")

    rclpy.init()
    node = Node('replay_calib_poses')

    from pymoveit2 import MoveIt2
    joint_names = ['J1_joint', 'J2_joint', 'J3_joint',
                   'J4_joint', 'J5_joint', 'J6_joint']

    moveit2 = MoveIt2(
        node=node,
        joint_names=joint_names,
        base_link_name='base_link',
        end_effector_name='flange',
        group_name='arm',
        use_move_group_action=True,
    )

    time.sleep(1.0)
    rclpy.spin_once(node, timeout_sec=0.5)

    print("\n按 ENTER 移动到下一位姿...")
    print("  到达后: 'r'=记录 'n'=跳过 'q'=退出")

    for i, rec in enumerate(records):
        if rec['joints'] is None:
            print(f"\n[{i}] 无关节角数据，跳过")
            continue

        joints = rec['joints']
        pos = rec['T_B_H'][:3, 3] if 'T_B_H' in rec else [0,0,0]
        e1 = rec.get('has_e1', False)
        e2 = rec.get('has_e2', False)
        n_pts = rec.get('n_pts', 0)

        print(f"\n{'='*50}")
        print(f"[{i}/{len(records)}] pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
        print(f"  e1={e1} e2={e2} pts={n_pts}")
        print(f"  关节角: {[f'{np.rad2deg(j):.1f}°' for j in joints]}")

        input("  按 ENTER 移动到该位姿...")

        # 用 MoveIt2 规划并执行
        traj = moveit2.plan(
            joint_positions=list(joints),
            joint_names=joint_names,
            tolerance_joint_position=0.02,
        )

        if traj is None or len(traj.points) == 0:
            print(f"  ❌ 规划失败，跳过")
            continue

        print(f"  执行中... ({len(traj.points)} 路径点)")
        moveit2.execute(traj)
        moveit2.wait_until_executed()
        time.sleep(0.5)

        # 等待用户确认
        print(f"  已到达。按 'r' 记录 / 'n' 跳过 / 'q' 退出: ", end='', flush=True)
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key == 'r':
                    print("  ✅ 记录")
                    # 用户需要在另一个终端按 auto_calib_node 的 'r'
                    break
                elif key == 'n':
                    print("  ⏭ 跳过")
                    break
                elif key == 'q':
                    print("  🛑 退出")
                    rclpy.shutdown()
                    return
            rclpy.spin_once(node, timeout_sec=0.1)

    print(f"\n回放完成！")
    rclpy.shutdown()


if __name__ == '__main__':
    main()
