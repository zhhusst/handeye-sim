# 双边角点主动手眼标定

本仓库只服务于两件事：

1. FANUC M-20iD/25、Gocator 线激光和平板角点的 ROS 2 仿真场景；
2. [完整方法原理 v1](线激光双边角点主动手眼标定_完整方法原理_v1.md) 的可测试实现。

方法主线是：

```text
人工给出首个双边可见位姿
→ 无手眼依赖的星形种子采集
→ 12-DOF-V2 变量投影联合初始化
→ 有限平板双边候选生成
→ 不确定性有效概率筛选
→ 手眼边缘信息增益选姿
→ 滚动求解与自适应停止
```

## 目录

```text
.
├── calibration_pipeline (位于 ROS Python 包内)
│   ├── seed_collection/   Phase 0b 纯算法组件
│   ├── solvers/           12-DOF-V2 求解器
│   ├── v2_backend/        角点投影、Jacobian、协方差、信息矩阵
│   ├── nbv/               有限平板候选、预测、有效性、评分、停止
│   ├── simulation/        无 ROS 的线激光几何与合成数据
│   └── pipeline.py        数据所有权和阶段状态机
├── ros2_ws/src/handeye_sim_bridge/
│   ├── handeye_sim_bridge/ ROS 话题、运动和可视化适配层
│   ├── config/             唯一运行参数
│   ├── launch/             ROS 2 启动文件
│   └── rviz/               RViz 配置
├── urdf/                   FANUC、Gocator 与场景模型
├── meshes/                 可视化网格
├── tests/                  ROS 无关的数值和单元测试
└── scripts/                构建、测试和仿真入口
```

详细的文档—代码对应关系和当前实验边界见
[实现状态](docs/IMPLEMENTATION_STATUS.md)。

## 快速验证

在开发容器内：

```bash
./scripts/test.sh
./scripts/run_core_demo.sh
```

构建 ROS 2 工作区：

```bash
./scripts/build.sh
```

启动现有 Gazebo + MoveIt 场景：

```bash
./scripts/start_simulation.sh
```

如果宿主机 X11 没有授权，启动脚本会自动切换到浏览器可视化。也可以显式启用：

```bash
./scripts/start_simulation.sh --web
```

然后打开
`http://localhost:6080/vnc.html?autoconnect=1&resize=scale`，Gazebo 和 RViz
会显示在同一个可交互桌面中。纯服务端验证仍可使用 `--headless`。

场景启动后，单独启动标定核心节点：

```bash
source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash
ros2 launch handeye_sim_bridge calibration_core.launch.py collect_seeds:=true
ros2 service call /bilateral_seed_collection/start std_srvs/srv/Trigger
```

Phase 0b 不读取手眼估计。启动服务前，操作者需要先把机器人调整到一个安全、稳定、
同时包含 `e_u` 和 `e_v` 的位姿。

## 开发约束

- `calibration_pipeline` 不得导入 `rclpy` 或 ROS 消息类型。
- ROS 节点只负责 I/O、TF、IK、轨迹执行与回退。
- 只有同时通过真实双边验证的观测才能加入标定数据集。
- 算法参数统一维护在
  [calibration.yaml](ros2_ws/src/handeye_sim_bridge/config/calibration.yaml)。
- 生成数据写入 `data/`，构建产物写入 `ros2_ws/{build,install,log}`，均不纳入版本控制。
