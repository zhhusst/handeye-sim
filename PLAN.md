# Num2 手眼标定仿真系统搭建计划

## 一、目标

在 ROS2 + Gazebo 中搭建 Fanuc + 线激光 + 平板的仿真环境，**可视化验证** plane_edge_9dof 采集流程和标定结果。

## 二、现有可复用的东西

| 来源 | 可用内容 |
|------|---------|
| `robot_description` | Fanuc M20-iD/25 URDF，含 6 轴连杆和关节限位 |
| `robot_control` | 机器⼈控制节点、FANUC 位姿格式转换（`transforms.py`） |
| `gocator_handeye_calib` | TF 框架（`base_link`→`flange`）、已有标定数据采集流程 |
| Num2 项目现有代码 | `nbv_edge_plane.py`（求解器）、`acquisition_sim.py`（采集仿真） |

## 三、系统架构

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Gazebo 仿真环境 │     │  RViz 可视化      │     │  标定节点        │
│                  │     │                  │     │                 │
│  Fanuc M20iD/25 │────▶│  3D 场景+TF      │◀────│  NBV 位姿规划   │
│  + 线激光模型    │     │  + 扫描线        │     │  + 数据采集      │
│  + 校准平板      │     │  + 断点          │     │  + LM 求解      │
│                  │     │                  │     │                 │
│  发布:           │     │  订阅 robot_state │     │  从单位阵出发    │
│  /joint_states   │     │  订阅 /laser_lines│     │  输出手眼结果    │
│  /laser_lines    │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘

TF 树:
  world → base_link → flange → laser_sensor
                           → calibration_plate
```

## 四、实施步骤

### Step 1: 创建仿真包 `calib_sim`（预计 1 天）

```bash
cd /home/z/research_contact_handeye/verification/Sim
# ROS2 包结构
calib_sim/
├── config/
│   └── calib_sim.yaml         # 仿真参数（FOV/噪声/板尺寸/初始手眼）
├── launch/
│   └── calib_sim.launch.py    # 启动 Gazebo + RViz + 标定节点
├── meshes/                    # 校准平板 STL (400×500×10mm 直角板)
├── urdf/
│   ├── calibration_plate.xacro  # 校准平板模型
│   └── calib_scene.xacro        # 完整场景（机器人+传感器+板）
├── worlds/
│   └── calib.world             # Gazebo world 文件
├── scripts/
│   ├── plate_visualization.py  # RViz Marker 显示扫描线和端点
│   ├── laser_sim.py           # 模拟线激光传感器（发布 scanline）
│   └── calib_node.py          # 标定主节点（位姿规划+采集+求解）
├── calib_sim/                 # Python 包
│   ├── __init__.py
│   ├── nbv_planner.py         # 从 nbv_edge_plane.py 移植
│   ├── solver.py              # 从 nbv_edge_plane.py 移植
│   └── acquisition.py         # 从 acquisition_sim.py 移植
├── setup.py / package.xml
└── README.md
```

**可复用的关键代码：**
- `solver.py` → 直接从 `nbv_edge_plane.py` 复制 `combined_solve_lm` / `combined_residuals`
- `acquisition.py` → 从 `acquisition_sim.py` 复制 `generate_linear_trajectory`
- `nbv_planner.py` → 复制 NBV 候选搜索逻辑

### Step 2: 激光传感器仿真（核心难点，预计 2 天）

Gazebo 原生没有线激光传感器型号。实现两种方案：

**方案 A（推荐，先用 Gazebo）：**
- 用 Gazebo 的 **Ray Sensor**（激光雷达 GMAPPING）模拟扫描线
- 在 `calibration_plate` 的碰撞体上返回测量点
- 只关注一条扫描线（xOz 平面），而非完整 3D 点云
- 发布为 `sensor_msgs/LaserScan` 或自定义消息

**方案 B（备选，纯数值可视化）：**
- 跳过 Gazebo 物理引擎
- 板面用 RViz Marker 显示
- 扫描线用已知板几何方程计算交点（和 now 的 `compute_fov_plate_scanline` 一样）
- 同时发布为 RViz Marker 做可视化
- 更快、更可控

**推荐先做方案 B** — 我们的核心目标是**验证位姿采集和标定流程**，不需要真实物理引擎。可视化用 RViz Marker 就够了。

### Step 3: 采集可视化验证（预计 1 天）

- 在 RViz 中显示：
  - 平板 3D 模型
  - 机器人模型（TF 树驱动）
  - FOV 三角形（Marker Line List）
  - 扫描线（Marker Points）
  - 端点（Marker Spheres，红色/蓝色标识）
- 运行 NBV 候选搜索 → 手动或自动遍历位姿序列
- 每帧校验：两边是否可见、端点归属是否正确

### Step 4: 全流程标定（预计 1 天）

- 从单位阵出发
- 采集 → 求解 → 输出结果
- 对比真值（仿真中已知手眼）
- 统计 R/t 误差

## 五、里程碑

| 天 | 完成内容 |
|----|---------|
| Day 1 | `calib_sim` 包基本结构 + URDF + launch file |
| Day 2 | 激光传感器仿真（方案 B 先做）→ RViz 能看到扫描线和端点 |
| Day 3 | 采集流程可视化 → 手动遍历位姿验证 `edge0/edge1` 归属 |
| Day 4 | 全流程自动标定 → 单位阵出发 → 出结果 |
| Day 5 | 留白（排错 + 方案 B→A 升级） |

## 六、依赖

```bash
# 已有
ros2 (humble)
gazebo (ignition)
RViz2

# 需要安装
sudo apt install ros-humble-xacro
sudo apt install ros-humble-joint-state-publisher-gui

# Python
pip install scipy  # FANUC WPR→矩阵转换（如果已有则跳过）
```

## 七、文件目录

```
/home/z/research_contact_handeye/verification/Sim/
├── PLAN.md                           ← 本文件
├── README.md                         # 使用说明
├── calib_sim/                        # ROS2 包
│   ├── config/calib_sim.yaml
│   ├── launch/calib_sim.launch.py
│   ├── urdf/calibration_plate.xacro
│   ├── urdf/calib_scene.xacro
│   ├── scripts/plate_visualization.py
│   ├── scripts/laser_sim.py
│   ├── scripts/calib_node.py
│   ├── calib_sim/__init__.py
│   ├── calib_sim/nbv_planner.py
│   ├── calib_sim/solver.py
│   ├── calib_sim/acquisition.py
│   ├── setup.py
│   ├── package.xml
│   └── README.md
├── Num2/                             # 现有代码软链接/拷贝
│   └── ... (nbv_edge_plane.py 等)
└── assets/                           # 模型文件
    └── plate.stl
```




你说得完全对，我之前的解释有错。让我重新来。

---

## 你的质疑拆开看

**质疑 1**：`C_S0`（e1/e2 中点）是激光线上的点，**不是**物理角点。

**→ 你对了。** 这是两个完全不同东西：

```
        板边1
         ╱
        ╱
  角点 C_B ●──────── 板边2
        ╱│
       ╱ │
  e1  ●  │              ← 激光线交到边上的两个断点
       ╲ │
        ╲│
         ●  e2
         │
    C_S0 =(e1+e2)/2      ← 这跟 C_B 是两回事
```

e1/e2 的中点只是激光线上两个断点的中点，它跟物理角点之间的距离取决于激光线切边的角度。

**质疑 2**：旋转后断点位置变了，中点也变了，怎么能要求 `C_S_new = C_S0`？

**→ 你也对了。** 我那个"固定 C_S"的约束是错的。中点不是不变量，硬要求它不变没有物理依据。

---

## 那什么是不变量？

**物理角点 `C_B`**。板不动，角点就不动。

问题是：**怎么从激光数据里估计 `C_B`？**

答案已经在你的 Phase 2a 数据里了。你有 4 帧不同平移的位姿，每帧都有 e1 和 e2。同一个物理边在不同帧里被激光切到不同的位置——这些 e1 点在基座标系里都落在同一条 3D 直线上（边 1）。e2 点也都落在边 2 上。两条边的交点就是 `C_B`。

---

## 正确算法：边线拟合 + 角点轨道

```
Phase 2a 4帧 → 拟合边1线、边2线（基座标系）→ 交点 = C_B →
  采样候选朝向 → t_BS' = C_B - R_BS'·C_S_ref → IK → 执行 →
    伺服微调（恢复到 e1+e2 均可见）→ 平移多样性
```

这里 `C_S_ref` 不再是 e1/e2 的中点，而是**你希望物理角点出现在传感器画面里的位置**——比如 FOV 正中央 `(0, 0, Z_mid)`。

关键差异：

| | 之前的错误理解 | 正确做法 |
|:--|:--|:--|
| 不变量 | `C_S`（中点）| `C_B`（物理角点）|
| `C_B` 来源 | 单帧 e1/e2 中点反算 | 多帧边线拟合求交点 |
| 约束 | `C_S_new = C_S0` | `C_S_new ≈ C_S_ref`（角点在 FOV 内即可） |
| 执行后 | 假设对准 | 伺服微调 |

---

## 关于板法向

不需要假设已知。两条边线都在板平面上，它们的叉积就是法向。Sharifzadeh 2020 的迭代 refine 可以进一步精确化——但初始估计可以从边线拟合直接得到。

---

要我实现吗？核心改动：
1. Phase 2a 结束后，用 4 帧 e1/e2 数据拟合两条边线 → 得 `C_B`
2. 旋转探索改用 `t_BS' = C_B - R_BS'·C_S_ref`（角点轨道）替代盲目 pitch/yaw

deepseek-v4-pro · 4% · ~