# Introduction 文献矩阵：字段说明与核验规则

本目录中的 `Introduction文献证据矩阵.csv` 是完整主表。它覆盖目录内全部 37 篇 PDF，并采用同一个 ID 与本文档后续的主题分组、文献演进链和论点—证据索引对应。

## 证据口径

- 页码统一写作 `PDF p.n`，指 PDF 阅读器中的页序，而不是期刊印刷页码；同时尽量给出章节。
- `Explicit` 表示作者在原文中明确陈述；`Inference` 表示根据流程图、算法步骤或实验过程作出的保守判断。
- `Not stated` 表示所检查的 Abstract、Introduction、Related Work、Discussion、Conclusion 及必要的方法/实验部分没有明确给出，不能据常识补全。
- `closed-loop feedback acquisition` 仅用于机器人下一步运动由在线传感器观测决定的工作。预先示教轨迹、离线计算路径、固定星形/随机运动和“程序自动执行”均不属于该类。
- `target imperfection jointly estimated` 仅用于把尺寸、点坐标、表面形貌或动态形变作为未知量放入标定估计的工作。滤波、选用高等级标定件、预先测量尺寸、拟合理想平面以及仅估计平面位姿均不算。
- 英文证据摘录均控制为短句；省略号表示从同一句中删去不影响含义的从句，不改变作者原意。

## Automation level 的固定枚举

1. `manual`
2. `semi-automatic`
3. `predefined automatic execution`
4. `model-based automatic planning`
5. `closed-loop feedback acquisition`

若一篇论文包含多个阶段，主表会按阶段分别说明，而不会用一个更强的标签覆盖全部流程。

## 关系标签

- `direct competitor`：与 RQ1 或 RQ2 的问题和实验对象高度重叠，Introduction 必须正面讨论。
- `predecessor`：构成我们方法某一关键思想的直接技术谱系。
- `supporting literature`：能支撑背景、动机或边界论点，但不与整套方法直接竞争。
- `peripheral`：已审阅，但对两个 RQ 的 Introduction 价值有限。

