# 法兰—Gocator 显示关系核查

## 结论

当前绘图中的 Gocator 外观模型采用与 RViz 相同的变换链：

$$
{}^{B}T_V = {}^{B}T_F\;{}^{F}T_{S,\mathrm{URDF}}\;{}^{S}T_V,
$$

其中：

- $F$：`fanuc_flange`；
- $S$：URDF/RViz 中的 `gocator_sensor` link；
- $V$：`Gocator_2450.dae` 网格坐标系；
- ${}^{B}T_F$：每个真实 seed 保存的法兰位姿；
- ${}^{F}T_{S,\mathrm{URDF}}$：URDF 固定关节；
- ${}^{S}T_V$：URDF visual origin。

## RViz 使用的两级固定关系

当前真机与仿真启动文件均把 `/workspace/urdf/calib_robot.urdf` 作为
`robot_description` 交给 `robot_state_publisher`。该 URDF 给出：

```xml
<joint name="fanuc_flange-gocator_sensor_joint" type="fixed">
  <origin xyz="-0.011579 -0.004621 0.359284"
          rpy="0.485145 0.160648 -1.509479" />
  <parent link="fanuc_flange" />
  <child link="gocator_sensor" />
</joint>
```

以及：

```xml
<link name="gocator_sensor">
  <visual>
    <origin xyz="0 0 -0.270" rpy="0 0 0" />
    <geometry>
      <mesh filename="file:///workspace/meshes/Gocator_2450.dae" />
    </geometry>
  </visual>
</link>
```

绘图程序在运行时直接解析这两个元素，没有在 Python 中重复硬编码数值。

焊枪使用同样的规则解析：

```text
fanuc_flange → weld_gun
xyz = [-0.046256, -0.000142, 0.375235] m
rpy = [-3.141540, -0.384130, -0.000070] rad
visual mesh = /workspace/meshes/weldgun.stl
mesh scale = [0.001, 0.001, 0.001]
```

## 六个法兰位姿核查

将本次 `seeds.json` 中六组关节角重新输入当前
`forward_kinematics_urdf()`，并与各 seed 保存的 `R_BF/t_BF` 比较：

| seed | FK—保存位姿旋转差 | FK—保存位姿平移差 |
|---|---:|---:|
| reference | 0.000000° | 0.0000 mm |
| ry_positive | 0.000000° | 0.0000 mm |
| ry_negative | 0.000000° | 0.0000 mm |
| rx_positive | 0.000000° | 0.0000 mm |
| rx_negative | 0.000000° | 0.0000 mm |
| rx_ry_positive | 0.000000° | 0.0000 mm |

这说明图中三色法兰轴和 RViz 中 `fanuc_flange` 的定义一致。

## 为什么不能用 calibration_result.json 摆放 RViz 外观模型

该次实验最终解中的 hand–eye 与当前 URDF 固定关节相比：

- 平移向量差：`[-14.591, -13.242, 13.920] mm`，模长约 `24.125 mm`；
- 旋转差：约 `177.701°`。

此外，真机轮廓适配器对 Gocator 原生工程坐标执行了
`coordinate_axis_sign: [1, -1, -1]` 的右手坐标归一化。求解器中的测量坐标定义
和 RViz 外观 link 因而必须明确区分。无论该次标定解为何落入当前分支，RViz
机身显示的唯一依据都应是 `robot_description` 中的固定关节，而不是离线求解结果。

绘图程序仍保留标定结果中的变换，用于解释 endpoint 和平板估计；但它不再参与
Gocator 外观模型的放置。元数据中同时记录了两套变换，避免再次混用。

## 运行时 TF 说明

核查时 ROS 图中没有正在运行的 `robot_state_publisher`，因此 `tf2_echo` 暂时找不到
这两个 frame。本次 RViz 关系核查来自 RViz 实际启动文件所加载的同一份 URDF。
环境启动后，可进一步现场核对：

```bash
ros2 run tf2_ros tf2_echo fanuc_flange gocator_sensor
```
