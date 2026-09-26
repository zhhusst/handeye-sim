# 旋转微步—平移伺服单步可视化

本目录对应论文自动采集方法中的一个真实连续过程：

1. **A**：`ry_positive` 分支中，旋转前的有效双边观测；
2. **B**：绕法兰局部 `+y_F` 纯旋转 `1°` 后、平移伺服前；
3. **C**：保持 B 的旋转姿态不变，执行一次双特征平移伺服后。

数据来自：

`/workspace/data/calibration_runs/20260820_144753_焊接书_位置1_真机_1`

它不是三个手工拼接位姿。A、B、C 是同一次真实 `ry_positive` 分支中连续记录的三个状态。

## 直接交互查看

```bash
cd /workspace
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py
```

### 四种显示模式

```bash
# 过程1：只显示 A→B，即局部 +y_F 旋转微步
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --process rotation

# 过程2：只显示 B→C，即保持姿态不变的平移伺服修正
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --process servo

# 完整过程：同时显示 A→B→C（默认模式）
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --process all

# 仅显示平板和绿色伺服目标区域
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --process target
```

如果没有显式指定 `--output`，四种模式分别保存为：

- `process1_rotation_microstep.png`；
- `process2_translation_servo.png`；
- `rotation_servo_microstep_current_view.png`；
- `servo_target_region_only.png`。

交互操作：

- 鼠标拖动：调整观察方向；
- 下方滑块：精确调整 elevation / azimuth / roll；
- `s`：保存当前视角的无滑块 PNG；
- `p`：在终端打印当前视角；
- `r`：恢复默认视角。

默认保存到：

`rotation_servo_microstep_figure/rotation_servo_microstep_current_view.png`

## 从原始 bag 重新提取 A/B/C

```bash
source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash
cd /workspace
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --refresh-from-bag
```

重新提取会生成/更新：

`rotation_servo_microstep_figure/microstep_states.json`

其中记录三个位姿的时间戳、关节角、法兰位姿、断点、扫描弦、A→B 旋转量和 B→C 平移量。

## 常用选项

```bash
# 无窗口直接输出静态图
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --save-only --elev 23 --azim -55 --roll 0

# 可选显示焊枪；默认隐藏，以突出 Gocator、激光平面和平板
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --show-weld-gun

# 隐藏激光平面，只保留平板上的三条真实扫描弦
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --hide-laser-planes
```

默认隐藏背景网格、三维坐标轴框和数值刻度，以保持论文图片干净。

## 平板上的绿色伺服目标区域

图中绿色半透明区域不是人为指定的矩形 ROI，而是把该次真机代码的双特征目标映射到平板局部坐标系后得到的**可接受扫描弦包络**：

- `|x_mid| <= 10 mm`；
- 两断点间距 `70 mm <= L <= 90 mm`；
- 绿色虚线为接近 `x_mid = 0、L = 80 mm` 的中心目标弦；
- 两条板边上的绿色粗线表示有效扫描弦端点允许落入的边界区间。

这里使用状态 C 的传感器位姿完成映射，因此它准确表示本次 B→C 平移伺服的等效目标区域，而不是对所有姿态都固定不变的世界坐标 ROI。C 只要进入绿色目标带即可停止，不要求精确压到绿色虚线上。

如需隐藏该区域：

```bash
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --hide-target-region
```

## 右上角传感器坐标系轮廓

右上角小图直接来自原始 bag 中的：

`/calibration/target_surface_points`

显示的是当前模式所含状态在传感器坐标系下的真实筛选后平板轮廓：

- `rotation`：叠加 A、B；
- `servo`：叠加 B、C；
- `all`：叠加 A、B、C；
- `target`：显示作为目标区域映射参考的 C。

横轴为 `x_S`，纵轴为 `z_S`，彩色圆点是对应的两个真实断点。小图不是根据直线方程重新生成的，而是由 bag 中的实际 PointCloud2 数据提取。

如需隐藏：

```bash
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --hide-profile-inset
```

## 在一个过程中选择具体位姿

使用 `--poses` 可以只显示当前过程中的一个或多个位姿。被选中的每个位姿会保留其完整信息：

- Gocator 实体模型；
- 法兰 RGB 坐标轴；
- 激光平面；
- 平板上的真实扫描弦和两个断点；
- 右上角对应的真实传感器坐标系轮廓。

只选择一个位姿时，Gocator 使用不透明实体显示；同时显示两个或三个重叠位姿时，Gocator 自动切换为半透明，以便区分不同状态。

示例：

```bash
# 旋转过程只显示旋转前的 A
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --process rotation --poses A

# 旋转过程只显示旋转后的 B
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --process rotation --poses B

# 伺服过程只显示修正前的 B
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --process servo --poses B

# 伺服过程只显示修正后的 C
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --process servo --poses C

# 完整过程中只比较 A 和 C
python3 rotation_servo_microstep_figure/visualize_rotation_servo_microstep.py \
  --process all --poses A C
```

未指定 `--output` 时，单独位姿图自动保存为 `selected_pose_A.png`、`selected_pose_B.png` 或 `selected_pose_C.png`。两个以上位姿则按组合命名，例如 `selected_pose_A_C.png`。

位姿必须属于所选过程：`rotation` 只允许 A/B，`servo` 只允许 B/C，`all` 允许 A/B/C；`target` 模式不包含位姿。

## 图中几何的来源

- 法兰位姿：bag 中 `/calibration/seed_motion_state` 的真实关节角，经当前 FANUC URDF 正运动学计算；
- Gocator/焊枪实体：当前 URDF 的固定关节与真实 mesh；
- 激光测量平面：当前标定结果中的 `^F T_S` 与两个实测断点共同确定；
- 平板：该次标定结果中的角点 `C` 和平板姿态；
- 彩色扫描弦：两个实测断点变换到基坐标系后，仅沿估计平板法向投影到板面；
- A→B、B→C 箭头：连接真实扫描弦中心；B→C 法兰箭头为真实平移。

注意：平板显示范围只是为了可视化观测角区域；其位姿、两条相邻边和扫描弦位置来自真实数据。
