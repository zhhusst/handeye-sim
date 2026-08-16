# FANUC + Gocator 真机安全接入指南

本指南适用于FANUC M-20iD/25、R-30iB和Gocator的首次标定联调。
真机后端分为`observe_only`、`plan_only`、`step_confirm`和`automatic`。
具备写权限的模式仍会以`DISARMED`启动，第二终端必须明确输入`ARM`才会
放行；第一次联调应使用`step_confirm`，不得直接跳到`automatic`。

## 1. 启动前

1. 机器人保持手动或T1模式，速度降到现场允许的安全值；作业人员在急停可及范围内。
2. 确认机器人IP和Gocator IP。默认分别为`192.168.0.10`和`192.168.0.19`。
3. 重建开发容器或安装`pycomm3>=1.2,<2`，然后执行`./scripts/build.sh`。
4. 不要启动旧`welding_robopath`的总launch，以免旧运动节点同时连接控制器。

## 2. 只观察启动

终端1：

```bash
cd /workspace
./scripts/start_environment.sh real --no-rviz \
  robot_ip:=192.168.0.10 sensor_ip:=192.168.0.19
```

启动时Gocator激光故意保持关闭。确认现场允许后，在终端2显式开启：

```bash
source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash
ros2 service call /gocator/set_laser_state \
  gocator_msgs/srv/SetLaserState '{enable: true}'
```

结束联调前关闭：

```bash
ros2 service call /gocator/set_laser_state \
  gocator_msgs/srv/SetLaserState '{enable: false}'
```

## 3. 必须留存的只读证据

```bash
ros2 service call /fanuc_joint_state/status std_srvs/srv/Trigger '{}'
ros2 service call /gocator_metric_adapter/status std_srvs/srv/Trigger '{}'
ros2 service call /measurement_sync/status std_srvs/srv/Trigger '{}'
ros2 service call /profile_endpoint_detector/status std_srvs/srv/Trigger '{}'
ros2 topic hz /gocator/profile
ros2 topic echo /fanuc/joint_states_raw --once
```

至少核对：Gocator原始值是毫米、适配后值是米；轮廓帧率稳定；
机器人静止时才产生`/calibration/flange_pose`；断点节点不读取仿真真值。
本机Gocator 2450的原生工程坐标在传感器前方为负Z；当前标定坐标系
通过右手变换`(x,y,z)->(x,-y,-z)`统一为前方正Z。这与历史精密球
标定结果中的旋转差异一致，不是为了让数值变正而任意翻转。

## 4. J2/J3约定验证

默认`j23_validated=false`，因此不会发布标定用`/joint_states`。用手持器
分别小角度单独改变J2和J3，同步记录：

- 手持器J1—J6；
- `/fanuc/joint_states_raw`；
- 控制器`CURPOS`或已知法兰位姿；
- 当前URDF/FK计算的法兰位姿。

分别比较`J3_raw`、`J3_raw + J2_raw`和`J3_raw - J2_raw`的FK结果。只有在
多个静止位姿上同一规则都与控制器一致，才可将`j23_factor`设为
`0`、`1`或`-1`，并将`j23_validated:=true`。

2026-08-13的四个真机位姿在UF1/UT1下得到：控制器关节角与EIP读数
六轴逐项一致；四组数据都唯一选择`j23_factor=1`。在加入已知
`base_link` Z向425 mm偏置后，正确模型的位置差为
0.00009–0.00019 mm、旋转差为0.00001–0.00003°；错误规则的位置差为
63–609 mm。因此已确认
`J3_URDF = J3_controller + J2_controller`，并开放只读的标准
`/joint_states`发布。运动权限由后文独立的软件门禁控制。

## 5. PC_TRACK_ALL STEP协议

示教器中的`PC_TRACK_ALL`保持原有结构，只把程序开头改为：

```text
UFRAME_NUM=1
UTOOL_NUM=1
```

本项目只使用该程序的STEP分支，不使用FIFO：

- `R[100]=0`：FIFO暂停；
- `PR[10]`：单个笛卡尔目标；EIP线格式写UF/UT为0/0，控制器读回会
  规范化为255/255；它不同于TP程序当前启用的UF/UT=1/1；
- `R[120]`：低速直线运动速度，默认5 mm/s；
- `R[110]=1`：唯一的运动触发；
- `R[102]=0/1/2`：空闲/运动/完成。

TP程序中没有定义`R[101]`确认或`R[103]`故障，因此新桥不会使用它们。
启动TP程序后，应先读状态；只有`R[100]=0、R[110]=0、R[102]∈{0,2}`
才允许软件解锁。

## 6. 双断点时序跟踪验证（不执行自动运动）

在开放自动种子运动前，先只验证原始轮廓上的双断点跟踪。终端1使用
`plan_only`启动环境并完成紫色ROI对齐；该模式不会写机器人运动目标：

```bash
cd /workspace
./scripts/start_environment.sh real --motion-mode plan_only
```

确认初始检测连续稳定后，在终端2锁定当前物理线段并启动种子跟踪模式：

```bash
source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash
ros2 service call /profile_endpoint_detector/lock std_srvs/srv/Trigger '{}'
./scripts/record_breakpoint_tracking_data.sh
```

先静止记录约10 s，再用示教器T1最低速度做连续、小幅、安全运动，最后再次
静止约10 s并按`Ctrl+C`结束记录。建议覆盖正负两个方向，但始终保持两个断点
可见。数据保存在`/workspace/data/breakpoint_tracking_runs/时间戳/`。记录脚本
会在保留2 s初始锁定轮廓后自动启停`SEED_TRACK_START/STOP`，但只改变感知模式，
不会解锁运动桥或发送机器人运动命令。结束时脚本会把检测器重置到初始紫色
ROI，避免下一次测试继承已经失效的局部跟踪引导。

状态服务中的`selection_mode=seed_temporal_track`表示已经使用双断点卡尔曼跟踪；
`temporal_mahalanobis`是两端点的创新距离，`temporal_missed_frames`是连续漏检
帧数，`temporal_search_radius_mm`是由协方差自适应得到的局部搜索半径。
`temporal_suspended=true`表示连续漏检或预测几何越界，检测器已停止盲目外推并
冻结到最后可信实测端点；此状态不会产生虚假的超长引导线。随后节点进入
第二级局部重捕获：搜索门限扩大到50 mm/12 mm，但候选相对最后可信线段的
长度变化不得超过20 mm、方向变化不得超过20°，并需连续3帧稳定才恢复跟踪。

## 7. 首次自动种子联调

先从逐步确认模式开始。终端1：

```bash
cd /workspace
./scripts/start_environment.sh real --motion-mode step_confirm
```

在示教器上以T1/低速启动`PC_TRACK_ALL`，使程序停留在循环中。终端2：

```bash
cd /workspace
./scripts/start_calibration.sh --backend real
```

选择“自动采集6个种子位姿”。完成初始轮廓对齐后，控制台会要求输入
`ARM`；在`step_confirm`中，每个目标还会显示最大关节变化、直线位移和
姿态变化，并逐步询问是否执行。任一步触发`R[110]`后，ROS软件撤防不能
中断正在执行的TP直线运动，必须使用FANUC的HOLD或急停。

逐步确认完成一整轮且核对实际运动方向后，才可把终端1改为：

```bash
./scripts/start_environment.sh real --motion-mode automatic
```

`automatic`仍要求第二终端每次运行前输入一次`ARM`，结束、失败或Ctrl+C时
自动软件撤防。

真机种子运动采用1°初始微旋转；`auto`模式仅在`|x_mid| <= 10 mm`且其余
静态余量充分时跳过动态预检。运动后先稳定0.8 s，再最多等待3 s的新鲜同步
双边帧。如果局部跟踪丢失但机器人已经回到已验证参考位姿，检测器会用紫色
模板ROI执行一次有界重捕获，再提交新的端点身份；它不会退回全轮廓盲搜。
日志出现`bilateral observation reacquired at the reference pose`表示这条恢复链
生效，而不是把一次丢帧误判成整次标定失败。

检测器区分两类先验：NBV未来观测使用较宽预测门限；回退到已经测量过的安全
位姿使用独立的`/calibration/detection_measured_prior`和窄跟踪门限。宽门限只
扩大候选召回范围，ROI内仍按斜率/高度突变重新分段，禁止把顶面、侧壁和工作台
合并拟合成一条线。

## 8. 人工采集回退

J2/J3验证后，启动不含自动运动的控制台：

```bash
./scripts/start_calibration.sh --backend real
```

人工采集保留为诊断回退，不再是真机主流程。操作者用手持器运动时，节点
仍只在机器人静止、时间偏差合格且双边断点有效时接受数据。

## 9. 自动模式放行门槛

只有以下条件全部满足，才由`plan_only`进入`step_confirm`，随后再考虑
`automatic`：

1. J2/J3、关节零位、方向和限位已经多位姿核对；
2. `base_link -> fanuc_flange`与控制器中的法兰位姿一致；
3. UTOOL、UFRAME、PR/R寄存器和TP程序版本已存档；
4. 当前桥已检查关节限位、最大步长、局部直线路径IK和CURPOS/FK一致性；
   尚无真机MoveIt碰撞场景，因此首次运行必须逐步确认并保持现场安全隔离；
5. 已验证FANUC HOLD/急停可中断TP运动；不要把ROS action cancel当作硬件停止；
6. `step_confirm`完整通过后，才允许使用`automatic`。

在TP程序归档和上述坐标、安全验证完成之前，
`/workspace/welding_robopath`仍作为只读参考库保留，不建议删除。
