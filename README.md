# 双边角点主动手眼标定

本仓库服务于三件事：

1. FANUC M-20iD/25、Gocator 线激光和平板角点的 ROS 2 仿真场景；
2. [完整方法原理 V6](线激光双边角点主动手眼标定_完整方法原理_v6.md) 的可测试实现；
   V6以当前代码为基线，V5保留为历史版本。
3. FANUC R-30iB与真实Gocator的分阶段、安全接入。

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
├── ros2_ws/src/handeye_calibration_core/
│   └── calibration_pipeline/ 纯算法，不依赖ROS和硬件
│       ├── seed_collection/ Phase 0b纯算法组件
│       ├── solvers/         12-DOF-V2求解器
│       ├── v2_backend/      Jacobian、协方差与信息矩阵
│       ├── nbv/             候选、预测、评分和停止
│       ├── simulation/      无ROS的线激光合成数据
│       └── pipeline.py      数据所有权和阶段状态机
├── ros2_ws/src/handeye_sim_bridge/
│   ├── handeye_sim_bridge/ 通用ROS标定节点与仿真兼容层
│   ├── config/             唯一运行参数
│   ├── launch/             ROS 2 启动文件
│   └── rviz/               RViz 配置
├── ros2_ws/src/handeye_sim_backend/ Gazebo场景、仿真真值与原始轮廓后端
├── ros2_ws/src/fanuc_m20id25_support/ 共用运动学与坐标约定
├── ros2_ws/src/fanuc_gocator_bridge/  真机只观察适配层
├── ros2_ws/src/gocator_profile_driver/ Gocator毫米原始轮廓驱动
├── ros2_ws/src/handeye_calibration_interfaces/ 后端无关运动接口
├── third_party/gocator_sdk/ 最小本地SDK运行集
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

统一环境入口为：

```bash
./scripts/start_environment.sh sim
```

旧命令`./scripts/start_simulation.sh`继续兼容。

如果宿主机 X11 没有授权，启动脚本会自动切换到浏览器可视化。也可以显式启用：

```bash
./scripts/start_simulation.sh --web
```

然后打开
`http://localhost:6080/vnc.html?autoconnect=1&resize=scale`，Gazebo 和 RViz
会显示在同一个可交互桌面中。纯服务端验证仍可使用 `--headless`。

场景启动后，在第二个终端打开中文交互式标定控制台：

```bash
cd /workspace
./scripts/start_calibration.sh
```

控制台会检查仿真环境，提示操作者在 RViz 中确认初始位姿，并提供自动采集、
人工采集六个种子、复用已有种子和查看最近结果四种入口。自动阶段会显示当前状态、
种子进度、目标失败次数、累计用时、NBV 候选、每次滚动求解结果及最终手眼矩阵。
确认初始位姿前，控制台会显示中点、工作距离、安全域余量、端点深度方向、关节余量
和四方向局部 IK。确认后系统实际执行局部 ±X/±Y 各 2° 的动态预检；不足 3/4
方向或未覆盖两个轴时会在移动前期明确拒绝该初始位姿。具体范围和调整方法见
[初始位姿指南](docs/INITIAL_POSE_GUIDE.md)。
端点物理身份始终固定，但两端的深度先后没有方向限制：初始验收使用
`|z(e2)-z(e1)| ≥ 15 mm`，正负两种观察姿态都可以进入动态预检。真实现场无需
肉眼看见空气中的激光平面，以Gocator二维轮廓、两个断点和终端数值反馈为准。
每次数据和日志保存在
`data/calibration_runs/<时间>/`，因此不会误用上一次运行的文件。

综合仿真噪声统一在
[calibration.yaml](ros2_ws/src/handeye_sim_bridge/config/calibration.yaml)
的 `simulation_noise` 中设置，包括轮廓与断点提取、机器人隐藏位姿误差、固定板面度、
同步抖动、离群点以及点/帧/断点漏检。距离、角度和时间单位分别为米、度和秒；修改后
必须重启仿真。标定控制台的环境检查会打印场景节点实际加载的数值。将该组内所有
标准差、延迟均值和概率设为 0 可建立理想数据基线，`random_seed` 保持固定则便于不同
算法版本使用同一噪声序列对比。完整数学定义见
[V6方法原理](线激光双边角点主动手眼标定_完整方法原理_v6.md#232-可配置综合噪声模型)。

仿真种子运动使用 5°分支目标、V6定义的 2°小步、0.45 秒轨迹和 0.15 秒
稳定等待；失去
控制器或新鲜轮廓时会在有限时间内明确失败，不再无限停在“机器人运动中”。
2026-07-28 的默认综合噪声 Gazebo 回归中，动态预检和 6 个种子用时 73.7 秒。
当前0.8 mm断点专项压力配置给每个种子采20帧、每个NBV采15帧。保持200 mm粗平移
初值的最近一次完整回归中，6种子78.1秒、5个NBV及初始化82.3秒，总计160.4秒。
交互默认5个NBV；选择第6个可能超过3分钟。人工设置
初始位姿的时间不计入。实际耗时仍会随初始位姿触发的
回退次数变化；不合格位姿会在动态预检阶段提前退出，不计作成功标定。

主动候选按 V6 的级联顺序执行平板可接近侧、IK、关节运动增量、鲁棒概率、信息增益、
目标状态碰撞和MoveIt连续路径检查；最终排序同时考虑信息增益与关节运动代价，避免
仅追求信息增益而选择腕部翻转或绕到基座另一侧的姿态。

候选规划使用内层安全域和额外名义余量，执行后的真实测量按外层硬有效域验收。某个
候选真实观测失败时，系统会丢弃该帧、回退到上一安全位姿、验证新鲜双边数据并拉黑
该候选后继续选择，而不会因一次可恢复失败终止整个标定。

主动标定节点已经改为后端无关：名义安装值来自配置，仿真真值只是可选评价器，
不会参与种子运动、12-DOF-V2残差或NBV评分。真机配置关闭真值评价，只报告
留出数据、重复性和协方差诊断。

## 真机接入状态

真机默认仍以`observe_only`启动：

```bash
./scripts/start_environment.sh real
./scripts/start_calibration.sh --backend real
```

该模式可以读取FANUC原始关节、Gocator毫米原始轮廓，完成米制转换、断点检测和
静止状态时间配对，但没有任何运动写入口。Gocator激光默认关闭。
J2—J3联动规则已经四个真机位姿验证为
`J3_URDF=J3_controller+J2_controller`，因此只读`/joint_states`已开放。
自动种子已接入现有`PC_TRACK_ALL`的STEP协议；首次真机运动必须用
`--motion-mode step_confirm`逐步确认，验证后才使用`automatic`。真机NBV执行仍要等
MoveIt碰撞环境和回退策略完成安全验收。

真机需要`pycomm3>=1.2,<2`。开发容器Dockerfile已声明该依赖；当前容器缺失时，
`start_real_environment.sh`会在启动任何节点前给出明确安装提示。
首次真机联调、激光开关、J2/J3验证与分阶段放行流程见
[真机安全接入指南](docs/REAL_HARDWARE_BRINGUP.md)。

仿真轮廓、物理断点和编码器法兰位姿使用同一触发时间戳严格配对；启用同步噪声后，
消息中法兰位姿会按配置从历史编码器快照选取，用于主动模拟内容层面的触发延迟。
当前每个物理种子采20个静止同步帧，对断点四坐标做整帧MAD筛选后聚合成一个求解
观测；原始帧按schema 3保留，并在进入NBV前执行真值无关的bootstrap稳定性门控。
每个NBV也以多帧作为一次事务提交，各帧保留各自法兰位姿，不能把多帧轮廓拼接后只
使用最后一帧位姿。2026-07-28 的默认综合噪声回归在 NBV 4 首次达到
0.0083°/0.0421 mm，NBV 5 为 0.0084°/0.0480 mm；这些数字只是固定随机种子的一次
仿真结果，不是多次鲁棒性统计或实物精度承诺。

## 开发约束

- `handeye_calibration_core/calibration_pipeline` 不得导入 `rclpy`或ROS消息类型。
- ROS 节点只负责 I/O、TF、IK、轨迹执行与回退。
- 仿真和真机后端必须满足同一米制轮廓、同时间戳法兰位姿和统一关节名契约。
- 真机运动权限按`observe_only → plan_only → step_confirm → automatic`开放。
- 只有同时通过真实双边验证的观测才能加入标定数据集。
- 算法参数统一维护在
  [calibration.yaml](ros2_ws/src/handeye_sim_bridge/config/calibration.yaml)。
- 生成数据写入 `data/`，构建产物写入 `ros2_ws/{build,install,log}`，均不纳入版本控制。
