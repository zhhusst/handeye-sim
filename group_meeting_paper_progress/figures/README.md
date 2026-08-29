# 图表说明与可追溯性

所有图均由 [analyze_progress_data.py](../scripts/analyze_progress_data.py) 生成，同时输出 PNG（300 dpi）和可编辑矢量 PDF。原始实验数据没有被修改。

## RQ1 图

### `rq1_seed_success_time_operations`

- 数据源：九组 `seed_collection.log`、`seeds.json`。
- 算法：从参考种子开始采集时间到 `seed collection complete` 作为自动种子耗时；统计日志中的 `dual-feature servo` 事件。橙色 D7 表示种子成功但后续平面初始化失败。
- 支持：九次均采满六种子；耗时中位 128.5 s。
- 限制：这不是与无反馈方法的时间对照。

### `rq1_per_target_time_and_servo`

- 数据源：九组 `seed_collection.log`。
- 算法：目标开始到该目标 `accepted physical seed` 的时间；同目标内纠偏事件计数。参考位姿没有主动旋转。
- 支持：定位最耗时/纠偏最多的分支，后续可据此优化运动顺序。
- 限制：目标之间还存在状态切换开销，热图各格之和不等于完整运行时间。

### `rq1_measured_pose_diversity`

- 数据源：九组 `seeds.json` 中实测 `R_BF`。
- 算法：计算 $\log(R_{0}^{T}R_i)$ 并转换为度；图显示旋转向量 x/y 分量。
- 支持：六种子在实测旋转空间中形成正负多轴激励，而不是重复姿态。
- 限制：二维图未显示较小的 z 分量；严格多样性数值以 `rotation_diversity` 字段为准。

### `rq1_feedback_timeline_D1`

- 数据源：D1 `full_run_bag` 的 `/calibration/seed_motion_state`。
- 算法：由真实端点重算 $x_{mid}$、$z_{mid}$ 和三维断点跨度；高频状态按最多 10 Hz 保留，状态/目标切换帧不降采样。
- 支持：直观显示主动旋转造成双特征偏离，随后平移纠偏重新进入软工作带；$z_{mid}$ 始终处于真机深度包络。
- 限制：曲线间断表示当时没有有效端点，而不是把缺失值插值；目标标签采用运行时状态机名称。

### `rq1_servo_convergence_D1`

- 数据源：D1 `seed_collection.log`。
- 算法：对每个非参考分支，按反应式纠偏事件顺序绘制纠偏前 $x_{mid}$ 和跨度。
- 支持：展示不同旋转轴的特征扰动和纠偏负担。
- 限制：日志只记录纠偏触发前的特征；最终进入软带后直接采集，因此终点不一定有一个额外日志点。

### `rq1_feedback_safety_all_runs`

- 数据源：九组 `full_run_bag` 的种子状态。
- 算法：每组分别绘制 $x_{mid}$、断点跨度、$z_{mid}$ 的箱线图；异常点不显示但参与汇总比例计算。
- 支持：跨九次运行检查真实反馈是否处于配置单位一致的工作域。
- 限制：箱线图隐藏离群点仅为版面清晰；硬门限外比例在 `rq1_summary.json` 中完整保留。

### `rq1_pipeline_success_funnel`

- 数据源：九组 `seeds.json`、`active_calibration.log`、`calibration_result.json`。
- 算法：分别统计六种子完成、平面初始化通过和最终结果文件存在。
- 支持：防止将 D7 的求解失败误记为自动采集失败。
- 限制：最终结果文件存在不等于外部精度达标。

## RQ2 图

### `rq2_abc_external_sphere_rmse`

- 数据源：`data/ablation_runs/shared_morphology_real9_r5/dataset_summary.csv`。
- 算法：绘制 5 次重复中位的固定刻字半径球 RMSE；D7 公共初始化失败为空。
- 支持：独立外部指标下，B 并未在每个数据集稳定优于 A；C 常出现严重过拟合。
- 限制：同一套球数据跨运行复用，前提是机械安装和球位置未改变。

### `rq2_ab_paired_internal_external`

- 数据源：同一消融目录的 `paired_a_b_differences.csv`。
- 算法：差值均定义为 B-A；绿色负值表示 B 更好。
- 支持：内部 surface RMS 在 8/8 改善，而外部球 RMSE 只有 4/8 改善。
- 限制：只有 8 个可解配对数据集，统计功效有限。

### `rq2_internal_residual_vs_external_error`

- 数据源：A/B/C `dataset_summary.csv`。
- 算法：横轴为标定数据 surface RMS，纵轴为独立球 RMSE；每点为一个数据集—模型组合的五次重复中位数。
- 支持：更低训练残差不能替代外部精度证据，尤其是 C 组。
- 限制：散点不能单独证明因果关系。

### `rq2_handeye_repeatability_abc`

- 数据源：A/B/C `dataset_summary.csv`。
- 算法：绘制五次帧 bootstrap 求解的旋转和平移离散度，纵轴为对数尺度。
- 支持：逐位姿形貌往往产生更不稳定的手眼解；B 对 A 没有一致重复性优势。
- 限制：这是同一批原始姿态内的帧 bootstrap，不是重新运动机器人的端到端重复标定。

### `rq2_estimated_shared_surfaces`

- 数据源：`trial_results.json` 中各可解数据集 repeat 0 的 B 组 7 个三阶 Legendre 系数。
- 算法：按代码中的归一化基函数，在 200 mm × 150 mm 名义板域上计算高度。
- 支持：展示共享形貌模型输出及不同标定物/位置的差异。
- 限制：**不是板面真值。** 未观测区域是模型外推；D6/D8 边缘较大幅值尤其不能在没有覆盖域与 CMM 数据时解释为真实翘曲。
