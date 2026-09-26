# 传感器坐标系下的轮廓演化图

该目录把原三维“旋转微步—平移伺服”图右上角的轮廓小图独立成论文级图片。数据仍来自同一次真实、连续的 `A → B → C` 过程：

- **A**：姿态激励前的有效双边观测；
- **B**：绕法兰局部 `+y_F` 旋转约 `1°` 后、平移伺服前；
- **C**：保持 B 的旋转姿态不变，完成紧随其后的一次平移伺服。

输入为：

`/workspace/rotation_servo_microstep_figure/microstep_states.json`

其中的轮廓来自真机 bag 的 `/calibration/target_surface_points`，断点来自同一时刻的 `/calibration/seed_motion_state`。绘图脚本没有重新拟合或合成轮廓。

## 生成图片

```bash
cd /workspace
python3 sensor_profile_evolution_figure/visualize_sensor_profile_evolution.py
```

默认生成两种排版，每种均提供 PDF、SVG 和 600 dpi PNG：

- `sensor_profile_evolution_triptych.*`：三联图，推荐用于正文；
- `sensor_profile_evolution_overlay.*`：三条轮廓叠加图，适合补充材料或汇报。

仅生成三联图：

```bash
python3 sensor_profile_evolution_figure/visualize_sensor_profile_evolution.py \
  --layout triptych
```

期刊排版通常由图注承担总标题，可使用：

```bash
python3 sensor_profile_evolution_figure/visualize_sensor_profile_evolution.py \
  --layout triptych --no-title
```

交互打开：

```bash
python3 sensor_profile_evolution_figure/visualize_sensor_profile_evolution.py \
  --layout triptych --show
```

## 字体说明

脚本优先使用 **Times New Roman**。当前系统没有安装微软专有的 Times New Roman 字体，因此当前已生成图片使用 Times 度量兼容的 **Nimbus Roman**。如果以后在系统中安装 Times New Roman，重新运行脚本会自动切换，无需改代码。数学字符使用与 Times 风格匹配的 STIX。

SVG 中的文字保持可编辑；PDF 使用可嵌入字体，适合论文排版。

## 图中数值

三联图使用完全相同的 `x_S-z_S` 坐标范围，避免各子图独立缩放掩盖真实变化：

- A：`x_mid = -2.15 mm`，`L = 80.74 mm`；
- B：`x_mid = -8.90 mm`，`L = 109.26 mm`；
- C：`x_mid = -9.25 mm`，`L = 89.12 mm`。

A→B 是约 `1°` 的纯姿态微步；B→C 是约 `9.00 mm` 的局部法兰平移，旋转变化仅约 `0.00016°`。因此图直接显示：姿态激励使传感器坐标系中的观测弦显著伸长，而闭环平移把断点间距重新带回 `70–90 mm` 的接受带。
