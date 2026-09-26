# 六种子星型采集位姿可视化

该脚本读取真机实验
`/workspace/data/calibration_runs/20260820_144753_焊接书_位置1_真机_1`
中的 `seeds.json` 与 `calibration_result.json`，把六个最终采集位姿统一画在机器人基坐标系中。

图中：

- 红、绿、蓝箭头分别表示每个位姿的法兰坐标轴 `x_F`、`y_F`、`z_F`；
- 半透明彩色实体是当前机器人 URDF 使用的真实 `Gocator_2450.dae` 外观模型；颜色与对应 seed 分支完全一致；
- 半透明深色实体是当前 URDF 使用的真实 `weldgun.stl` 焊枪模型；
- 虚线从参考位姿分别连接到五个目标位姿，用于强调星型采集关系；
- 半透明浅黄色平面表示由本次标定结果得到的板角局部区域。其位姿来自数据，显示尺寸仅用于辅助观察。

程序不是把模型简单放到标定求得的测量坐标原点，而是直接解析
`/workspace/urdf/calib_robot.urdf` 中两级变换：

1. `fanuc_flange-gocator_sensor_joint`：RViz 使用的法兰到传感器 link 固定关节；
2. `gocator_sensor/visual/origin`：传感器 link 到外观网格的变换，其中包含
   `xyz="0 0 -0.270"` 的模型原点偏移。

因此图中的 Gocator 机身与法兰的关系和当前 RViz/robot_state_publisher 一致。
`calibration_result.json` 中的 hand–eye 仍用于解释标定观测与平板结果，但不再用于摆放外观模型；这两个坐标定义不能混用。

详细的数值核对、FK 一致性结果和两套坐标定义的区别见
[`TRANSFORM_CHECK.md`](./TRANSFORM_CHECK.md)。

焊枪默认显示，并同样直接使用 URDF 中
`fanuc_flange-weld_gun_joint` 的固定关系。为了避免六把焊枪相互遮挡法兰坐标轴，
绘图时采用半透明显示。若当前论文图不需要焊枪，可关闭：

```bash
python3 /workspace/star_seed_pose_figure/visualize_star_seed_poses.py \
  --hide-weld-gun
```

## 位姿颜色对应关系

Gocator 与对应位姿的分支标记、虚线和编号共用颜色：

| 编号 | 位姿 | Gocator/分支颜色 |
|---:|---|---|
| 0 | Reference | 黑色 |
| 1 | $R_{y_F}(+)$ | 黄色 |
| 2 | $R_{y_F}(-)$ | 浅蓝色 |
| 3 | $R_{x_F}(+)$ | 绿色 |
| 4 | $R_{x_F}(-)$ | 粉色 |
| 5 | $R_{x_F,y_F}(+,+)$ | 橙色 |

Gocator 使用半透明表面，因此六个位姿的实体互相重叠时，仍能观察后方模型及其对应关系。红、绿、蓝箭头只表示每个法兰自身的 $x_F,y_F,z_F$ 轴，不表示位姿身份。

## 安装变换辅助线

法兰原点到 `gocator_sensor` 原点的灰线，以及传感器原点沿 $+z_S$ 的黑色短线，默认均不显示。它们只是变换诊断标记，不是实体、线缆或激光线。

如需临时核对安装变换，可显式打开：

```bash
python3 /workspace/star_seed_pose_figure/visualize_star_seed_poses.py \
  --show-transform-guides
```

## 交互查看

在能够正常打开桌面窗口的终端中运行：

```bash
python3 /workspace/star_seed_pose_figure/visualize_star_seed_poses.py
```

操作方式：

- 鼠标拖动：自由旋转三维视角；
- 鼠标滚轮或右键拖动：缩放，具体方式取决于 Matplotlib 后端；
- 底部三个滑块：精确调整 elevation、azimuth 和 roll；
- `p`：在终端打印当前视角参数；
- `s` 或 `Save PNG`：按当前视角保存无标题、无网格的论文图片；
- `r` 或 `Reset view`：恢复默认视角。

交互窗口和保存图片默认均不显示三维背景网格，以保持论文图干净。如需临时查看空间网格，可启动时增加：

```bash
python3 /workspace/star_seed_pose_figure/visualize_star_seed_poses.py \
  --show-grid
```

机器人基坐标系的三维外框、`x_B/y_B/z_B` 名称和数值刻度也默认全部隐藏。若需要检查绝对空间坐标，可临时恢复：

```bash
python3 /workspace/star_seed_pose_figure/visualize_star_seed_poses.py \
  --show-base-axes
```

`--show-grid` 会自动保留基坐标轴外框和刻度，否则背景网格没有可用的刻度位置。

默认保存到：

```text
/workspace/star_seed_pose_figure/star_seed_poses_current_view.png
```

同时生成 `star_seed_pose_data.json`，记录六个位姿、传感器位姿、关节角、相对参考位姿的旋转和平移以及保存图片的观察角度。

## 按给定角度直接导出

无需打开窗口即可生成静态图片：

```bash
python3 /workspace/star_seed_pose_figure/visualize_star_seed_poses.py \
  --save-only --elev 24 --azim -58 --roll 0 \
  --output /workspace/star_seed_pose_figure/star_seed_poses_preview.png
```

可通过 `--axis-length` 调整法兰坐标轴长度，通过 `--hide-labels` 隐藏位姿名称，通过 `--hide-board` 隐藏平板局部面。`--mesh-voxel-mm` 控制 Gocator 网格的简化尺度，默认 2 mm；`--weld-gun-voxel-mm` 单独控制焊枪网格，默认 5 mm。数值越小外观越精细，但交互旋转越慢。完整参数可运行：

```bash
python3 /workspace/star_seed_pose_figure/visualize_star_seed_poses.py --help
```

## 如果以 root 运行时窗口无法打开

若终端出现 `No protocol specified`，说明 root 无权连接当前桌面会话。可以切换到登录桌面的用户运行：

```bash
sudo -u z env DISPLAY=:0 XAUTHORITY=/home/z/.Xauthority \
  python3 /workspace/star_seed_pose_figure/visualize_star_seed_poses.py
```

若实际 `DISPLAY` 不是 `:0`，应把上面的值替换为桌面终端中 `echo $DISPLAY` 的输出。
