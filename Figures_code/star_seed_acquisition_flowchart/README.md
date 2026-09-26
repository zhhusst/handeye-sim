# 六种子星型自动采集流程图

本目录用于生成论文中的“自动六种子星型采集流程”图。它不会启动 ROS，也不会修改任何真机代码。

## 生成命令

```bash
cd /workspace
python3 star_seed_acquisition_flowchart/draw_star_seed_acquisition_flowchart.py
```

默认同时生成中英文版本，每个版本均包含可编辑矢量 `SVG`、论文排版用 `PDF`、预览用 300 dpi `PNG` 和 Graphviz 源文件 `DOT`。

只生成中文版：

```bash
python3 star_seed_acquisition_flowchart/draw_star_seed_acquisition_flowchart.py --language zh
```

## 图中流程的代码依据

- 控制台初始位姿确认和检测器锁定：`scripts/calibration_console.py::_confirm_initial_pose`
- ROI/CSRT 双断点锁定与跟踪：`profile_endpoint_detector_node.py::_lock_callback`，以及 `calibration_pipeline/roi_tracking/breakpoint_pipeline.py`
- 星型分支顺序：`calibration_pipeline/seed_collection/rotation_scheduler.py::star_rotation_plan`
- 微步旋转、反馈判断、回退和参考位姿恢复：`seed_collection_node.py::_issue_micro_rotation`、`_after_micro_rotation`、`_rollback`、`_return_reference`
- 双特征伺服和 Broyden 更新：`translation_servo.py::BroydenDualFeatureServo`
- 定点多帧采集、MAD 内点筛选和种子保存：`seed_collection_node.py::_begin_seed_capture`、`_try_finish_seed_capture`、`_save_seed_batch`

## 与所选真机数据的对应关系

输入数据：

`/workspace/data/calibration_runs/20260820_144753_焊接书_位置1_真机_1`

流程图脚本会自动读取该目录的 `seeds.json` 和 `seed_collection.log`。本次运行的事实为：

- 不是“6 个旋转分支”，而是 **1 个静止参考位姿 + 5 个旋转分支**；
- 实际顺序为 `reference → ry_positive → ry_negative → rx_positive → rx_negative → rx_ry_positive`；
- 6 个物理位姿最终分别保留 `18, 16, 14, 14, 18, 18` 个同步内点帧，共 98 帧；
- `ry_negative` 和 `rx_ry_positive` 分支实际触发过硬保护回退与微步缩小，因此恢复路径不是臆造的；
- 独立动态预检在本次运行中为 `off`，所以图中把它标为可选工程模块，不放进本次主执行路径；
- 六种子完成后才进入共享模型 12-DOF-V2 初解与 NBV，图中只保留接口，不展开 NBV。

## 论文使用建议

- 中文论文或答辩使用 `star_seed_acquisition_flowchart_zh.pdf`；
- 英文投稿使用 `star_seed_acquisition_flowchart_en.pdf`；
- 若版面较窄，建议在排版软件中裁去右侧“公式”和“本次运行证据”两个注释框，但不要裁掉红色回退路径，因为它体现闭环采集与预定义轨迹执行的本质区别。
