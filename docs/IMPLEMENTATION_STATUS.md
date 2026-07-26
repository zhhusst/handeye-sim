# 方法 v1 实现状态

本文档以仓库根目录的《线激光双边角点主动手眼标定_完整方法原理_v1》为唯一设计基线。

| 方法阶段 | 实现位置 | 状态 |
|---|---|---|
| Phase 0a 粗手眼初值 | `calibration_pipeline.cli`、仿真配置 | 已实现扰动入口；实机值由上层提供 |
| Phase 0b 星形旋转计划 | `seed_collection/rotation_scheduler.py` | 已实现 |
| 双边特征与 ROI 硬约束 | `seed_collection/features.py` | 已实现 |
| 端点连续身份 | `seed_collection/endpoint_tracker.py` | 已实现；仿真优先使用场景发布的物理标签 |
| 无标定试探平移伺服 | `seed_collection/translation_servo.py` | 已实现并由 ROS 节点执行 |
| 失败回退与步长减半 | `seed_collection_node.py` | 已实现 |
| 种子旋转多样性 | `seed_collection/seed_observability.py` | 已实现 |
| 12-DOF-V2 变量投影 | `solvers/twelve_dof_v2.py` | 已实现 |
| 角点解析投影 | `v2_backend/corner_projection.py` | 已实现 |
| 约化 Jacobian、协方差和 Schur 信息 | `v2_backend/` | 已实现 |
| 有限平板恢复与四边分类 | `nbv/finite_board_intersection.py` | 已实现 |
| 指定相邻边候选生成 | `nbv/candidate_generator.py` | 已实现 |
| 固定法兰命令下的未来轮廓 | `nbv/profile_predictor.py` | 已实现 |
| 9 维 Sigma 点有效概率 | `nbv/validity.py` | 已实现 |
| 完整变量投影信息增益 | `nbv/scoring.py` | 已实现 |
| 自适应停止 | `nbv/stopping.py` | 已实现 |
| IK、关节限位 | ROS 运动适配层 | Phase 0b 已接入 |
| NBV 碰撞与整条路径检查 | MoveIt 适配层 | 尚未接入自动流水线 |
| 完整 ROS NBV 自动执行 | 待新增硬件适配节点 | 核心决策已实现，运动闭环待实验验证 |

“已实现”表示模块、数值约束和单元测试存在，不表示文档第 20 节中的实验参数已经得到证明。
尤其是 Sigma 点概率阈值、信息增益阈值、碰撞检查和真实 Gocator 断点鲁棒性，仍需实验标定。

## 重要边界

种子采集节点只负责 Phase 0b，不会偷偷使用 URDF 手眼真值。候选评分模块始终对一个固定
的法兰命令传播状态不确定性。NBV 候选加入信息矩阵时，会对增广数据重新执行角点变量投影，
而不是把一个孤立的候选 Jacobian 简单相加。
