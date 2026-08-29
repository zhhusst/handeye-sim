# 统计结果文件

- `rq1_run_summary.csv`：九组运行级种子、初始化、NBV 和结果状态。
- `rq1_target_summary.csv`：54 个物理种子的逐目标时间、纠偏、旋转命令和帧内点数。
- `rq1_servo_events.csv`：360 个反应式纠偏触发前的双特征记录。
- `rq1_rotation_vectors.csv`：54 个种子相对参考位姿的实测旋转向量。
- `rq1_seed_state_timeseries.csv`：九个 MCAP 包中种子状态的至多 10 Hz 降采样时间序列；原始状态切换被保留。
- `rq1_summary.json`：RQ1 汇总数字，包括成功率、耗时、安全域比例、回退和 NBV 数。
- `rq2_abc_dataset_summary.csv`：A/B/C 可用数据集的五次重复中位结果。
- `rq2_ab_paired_differences.csv`：B-A 逐数据集配对差值，负值表示 B 对误差指标更优。
- `当前工作完成度表.md`：按证据等级盘点代码、数据和实验缺口。
- `方法代码与证据索引.md`：将 PPT 方法内容映射到生产代码和真机证据。

所有 CSV 使用 UTF-8 BOM，便于 Excel 正确显示中文路径。原始来源路径写入每行 `dataset` 字段，并可由分析脚本中的固定 D1–D9 映射追溯。
