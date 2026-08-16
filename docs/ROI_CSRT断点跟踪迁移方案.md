# ROI/CSRT 断点跟踪迁移方案（已评审、已实现）

- 状态：**已实现并完成离线回放验证**（2026-08-16）
- 适用阶段：真机初始对齐与六种子自动采集
- 当前策略：仿真默认 `classic`，真机默认 `roi`
- 离线依据：`20260814_140129_loss_rollback_0.mcap`

---

## 一、评审结论

迁移方向可行，但不能简单理解成“把 Kalman 换成 CSRT”。正确结构是：

1. CSRT 只跟踪两个断点周围的局部图像区域，提供下一帧的关联范围；
2. 真正发布给标定求解器的断点和平板表面点，仍然必须从当前帧原始二维轮廓中计算；
3. `CSRT.update()` 返回成功不代表断点有效，还必须检查双 ROI、断点位置、断点间距和当前表面几何；
4. 机器人回退后不能恢复旧 CSRT 对象，而要在机器人到位后的当前图像中，用已保存的物理断点重新初始化 CSRT；
5. 单帧失败只记为瞬态拒绝，连续失败才进入 LOST；LOST 后必须能够通过已测断点先验或参考位姿快照重建跟踪器。

因此，ROI/CSRT 是“时域关联后端”，不是替代二维轮廓几何测量的黑盒视觉检测器。

---

## 二、原方案中需要修正的地方

### 2.1 在线节点原先没有调用 `tracking_pipeline.py`

原文认为在线系统由 `tracking_pipeline.py` 统一驱动，这与实际代码不符。原在线状态机直接写在：

```text
handeye_sim_bridge/profile_endpoint_detector_node.py
```

`perception/tracking_pipeline.py` 是零 ROS 离线复现版本。迁移时必须显式把 ROI 后端接入在线节点，不能只修改离线 pipeline。

### 2.2 20 mm ROI 不是已经证明的固定最优尺寸

20 mm 可以作为初值，但 CSRT 需要看到断点周围的结构上下文。如果 ROI 太小，只包含一小段近似直线，两个跟踪器容易同时收敛到内部相似结构。

当前实现保留 `roi_size_m=0.020` 为可配置初值，同时增加：

- ROI 核心区约束；
- ROI 中心跳变约束；
- 断点跳变约束；
- 相对初始弦长约束。

指定 loss/rollback bag 中，单纯运行两个 20 mm CSRT 时，OpenCV 在全部 897 个后续帧都返回了 tracker success，但只有前 171 帧通过断点几何验证。这证明 `tracker success` 绝不能直接作为系统有效观测。

### 2.3 “恢复 ROI 快照”不能等同于恢复 CSRT 内部状态

OpenCV CSRT 对象无法可靠序列化，也不应该把运动前的图像模板直接恢复到运动后的帧。真正可恢复的是已验证机器人位姿对应的物理断点 E1/E2。

回退流程必须是：

```text
发布已测断点先验
→ 机器人回退
→ 当前轮廓重新接近断点先验
→ 连续若干帧几何检测稳定
→ 在当前图像上重新生成 ROI 并初始化两个 CSRT
```

现有种子节点已经发布 `/calibration/detection_measured_prior`，因此不需要新增 `/calibration/detection_snapshot` 话题，也不需要复制一套 ROI 快照进种子节点。

### 2.4 不能每个 VALID 视频帧都覆盖机器人回退快照

检测节点每帧有效，只说明当前图像跟踪有效；它不一定对应种子状态机保存的那个安全机器人位姿。如果检测节点自己不停覆盖“last valid snapshot”，可能发生“机器人回退到 A 位姿，但检测器恢复了 B 帧 ROI”的错配。

当前实现复用种子节点在 `_remember_last_valid()` 时保存的物理端点，并通过 measured-prior 传给检测节点，从而保证机器人回退目标和检测恢复锚点一致。

### 2.5 多帧失败不再与旧 `lost_frames` 重复计数

原方案“连续失败 3 帧后才给 lost_frames 加一，再等 5 次进入 LOST”语义重复，实际需要 7 帧以上才进入 LOST。

当前语义是：

- 第 1～2 个连续失败帧：`REJECTED`；
- 第 3 个连续失败帧：进入 `LOST`；
- 任意有效帧：连续失败计数清零。

ALIGN 的稳定计数只比较相邻的“有效观测”，中间没有测量的坏帧既不增加也
不清空计数；下一组有效断点如果真的发生位移，才按稳定阈值重新计数。lock
仍要求服务调用时最新一帧为 VALID，避免在坏图像上初始化 CSRT。

失败帧不发布旧断点，避免把运动中的陈旧测量写入静态种子数据。

### 2.6 PREDICTED_TRACK 不能在未来位姿尚未到达时初始化 CSRT

NBV 先验描述的是未来视角下的物理断点。收到先验时，当前图像中通常还不存在该目标位置。正确流程是：

```text
收到未来断点先验
→ 保存先验并进入待重获取状态
→ 机器人运动
→ 当前轮廓与未来先验接近
→ 连续稳定确认
→ 使用到位后的当前图像初始化 CSRT
```

不能收到先验后立刻在运动前图像的未来 ROI 中初始化 CSRT。

---

## 三、最终架构

```text
profile_endpoint_detector_node.py
│
├── backend=classic
│   └── 原 ProfileEndpointDetector + DualEndpointKalmanTracker
│
└── backend=roi
    └── ROIBreakpointPipeline
        ├── ALIGN：紫色模板约束 → 轮廓几何检测 → 自动建立双 ROI
        ├── TRACK：CSRT×2 → 当前轮廓重算断点/表面 → 联合验证
        ├── LOST：以最近已验证物理断点为锚点局部重获取
        └── PREDICTED_TRACK：以 NBV 未来断点为到位后重初始化锚点
```

对外接口保持不变：

```text
输入： /gocator/profile
输入： /calibration/detection_prior
输入： /calibration/detection_measured_prior
输入： /calibration/detection_control

输出： /calibration/endpoints
输出： /calibration/target_surface_points
输出： /calibration/detection_guide
输出： /profile_endpoint_detector/diagnostics
```

因此 `seed_collection_node.py`、求解器和 NBV 不需要因为跟踪后端切换而改变测量消息格式。

---

## 四、状态机细节

### 4.1 ALIGN

1. 操作人员把黄色原始轮廓中的平板表面移到紫色模板附近；
2. 系统从模板中心对应的 X 坐标出发，沿有序轮廓向左右生长；
3. 当新点对当前平板直线的残差超过阈值时，认为到达物理边界；
4. 检查断点与紫色模板的端点距离、法向距离和角度差；
5. 连续 `minimum_lock_frames` 帧稳定后允许 lock；
6. lock 时以两个实测断点为中心建立 ROI，并在当前图像中初始化两个 CSRT。

ALIGN 仍是 bounded detection，不允许在整个工作台轮廓中无约束寻找最长线段。

### 4.2 TRACK

每帧执行：

1. 使用固定物理坐标映射把二维轮廓光栅化；
2. 两个 CSRT 分别更新两个 ROI；
3. 以两个 ROI 中心的平均 X 为平板内部起点，重新从原始轮廓向两侧生长；
4. 得到当前实测断点和平板表面点；
5. 将断点与两个 ROI 做最小代价关联，保持 E1/E2 身份；
6. 检查 ROI 核心区、ROI 跳变、断点跳变、绝对弦长和相对弦长；
7. 通过后才发布当前测量。

### 4.3 LOST 与回退恢复

TRACK 连续失败达到阈值后进入 LOST，并冻结最后一组可信物理断点。

```text
seed_collection_node
  → 发布 last_valid_feature 的两个实测断点
  → 命令机器人回到 last_valid_joints

ROI detector
  → 把实测断点设为 reacquire_anchor
  → 不在运动途中初始化 CSRT
  → 等当前原始轮廓回到 anchor 附近
  → 连续稳定后在当前帧重新初始化 CSRT
```

返回参考位姿时，检测器使用 lock 时内部保存的 reference endpoints，语义相同。

### 4.4 PREDICTED_TRACK

NBV 预测端点只作为未来到位后的几何锚点。机器人到位且轮廓与预测相符后才初始化 CSRT。预测取消时，使用运动前的实测断点作为恢复锚点。

---

## 五、光栅化约束

### 5.1 固定坐标映射

在一个跟踪会话内，X-Z 到像素的映射保持固定：

\[
u=\frac{x-x_{\min}}{r},\qquad
v=\frac{z_{\max}-z}{r}.
\]

每帧单独缩放到自身包围盒会抹掉真实运动，不能使用。

### 5.2 在线范围建立

lock 时根据当前完整原始轮廓的稳健范围加 `raster_margin_m` 建立固定光栅。若重获取时轮廓已经超出范围，则在重初始化 CSRT 前重建光栅。

### 5.3 越界点

越界轮廓点必须丢弃，不能裁剪到图像边缘。否则大量越界点会形成一条虚假的高亮边界，CSRT 可能跟踪这条假线。

---

## 六、实现文件

### 新增

```text
calibration_pipeline/roi_tracking/breakpoint_pipeline.py
scripts/evaluate_roi_breakpoint_tracker_bag.py
tests/test_roi_breakpoint_pipeline.py
```

### 修改

```text
calibration_pipeline/roi_tracking/rasterizer.py
calibration_pipeline/roi_tracking/__init__.py
handeye_sim_bridge/profile_endpoint_detector_node.py
handeye_sim_bridge/config/calibration.yaml
fanuc_gocator_bridge/config/real_calibration.yaml
handeye_calibration_core/package.xml
```

仿真默认：

```yaml
endpoint_detection:
  backend: classic
```

真机覆盖：

```yaml
endpoint_detection:
  backend: roi
```

这样先在真机六种子问题上验证 CSRT，不改变已经跑通的仿真基线。

---

## 七、主要配置

```yaml
endpoint_detection:
  backend: roi
  tracker_name: csrt
  roi_size_m: 0.020
  fail_streak_frames: 3
  raster_res_mm: 0.25
  roi_process_every_n_frames: 4
  raster_point_radius_px: 2
  raster_margin_m: 0.020
  raster_maximum_dimension_px: 2400
  core_fraction: 0.70
  roi_jump_m: 0.015
  bp_jump_m: 0.015
  roi_tracking_minimum_chord_ratio: 0.45
  roi_tracking_maximum_chord_ratio: 2.50
  plate_growth_residual_threshold_m: 0.0008
  roi_surface_residual_threshold_m: 0.0012
```

ROI 尺寸仍需通过更多自动种子 bag 做消融，不能仅凭一个 bag 宣称 20 mm 最优。

真机设为每 4 帧处理一次，是为了与已经验证的离线命令
`--stride 4` 保持一致，并抑制 Gocator 原始高频流中的周期性坏帧。被跳过
的帧不会发布空断点，因此不会被种子节点误判为检测失败。

---

## 八、验证结果与边界

### 8.1 单元测试

已覆盖：

- ALIGN 稳定后自动生成双 ROI；
- 单帧失败不进入 LOST；
- 连续三帧失败进入 LOST；
- measured prior 驱动回退后重新初始化；
- reference snapshot 使用锁定时实测断点，而不是紫色模板端点。

### 8.2 指定 MCAP 离线结果

```bash
python3 scripts/evaluate_roi_breakpoint_tracker_bag.py \
  /workspace/data/breakpoint_tracking_runs/20260814_140129_loss_rollback/20260814_140129_loss_rollback_0.mcap \
  --stride 4
```

该 bag 包含人为丢失和回退过程。统一按 `stride=4` 回放得到：

| 状态机 | VALID/总帧 | LOST→恢复 | 最终模式 |
|---|---:|---:|---|
| classic（原几何检测/Kalman状态机） | 159/898（17.7%） | 0/1 | LOST |
| ROI/CSRT | 630/898（70.2%） | 2/2 | TRACK |

ROI/CSRT 平均处理时间为 41.6 ms/处理帧，最大连续无观测为 137 帧；后者对应 bag 中目标实际离开视野的区间。结果文件保存在同一实验目录的 `roi_csrt_evaluation.json`。

该 bag 中确实存在目标完全消失区间，因此不能用全程 acceptance rate 单独判断跟踪器好坏；必须同时报告 LOST 次数、恢复次数、有目标区间有效率、最大连续无目标帧数和平均处理时间。

### 8.3 尚不能声称完成的验证

离线结果证明“机制可接入且能够恢复”，但还不能替代真机闭环测试。下一次真机测试必须验证：

1. ALIGN 后自动 ROI 是否与手动 CSRT ROI 的语义一致；
2. 自动微旋转、突然停止、反向回退时是否保持身份；
3. 第一个种子失败后能否自动回到 last-valid 并继续；
4. 六种子是否能够完整采集；
5. 在线处理帧率是否满足静态多帧采集超时限制。

只有六种子自动流程实际跑通后，才能把 ROI/CSRT 标记为真机稳定方案。
