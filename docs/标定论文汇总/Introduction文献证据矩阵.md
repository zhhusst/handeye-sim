# 面向 Introduction 写作的标定文献证据矩阵

> 研究范围：机器人末端安装的 2D laser profiler / line-structured-light sensor 手眼标定。  
> RQ1：未知准确手眼时，如何仅靠二维轮廓反馈自主获得有效且具有姿态多样性的标定数据？  
> RQ2：普通平板不是理想数学平面时，如何用跨位姿共享形貌避免固定板面误差污染手眼外参？

完整的 28 字段逐篇主表位于 [Introduction文献证据矩阵.csv](Introduction文献证据矩阵.csv)，覆盖本目录全部 37 篇 PDF。CSV 是权威、可筛选版本；本文档提供可读索引和 Introduction 所需的综合推理。字段、页码和证据等级的口径见 [Introduction文献矩阵_字段说明.md](Introduction文献矩阵_字段说明.md)。

## 1. 主表阅读索引

下表是完整主表的阅读入口。`实时反馈`仅指观测在线决定后续机器人运动；“自动求解”和“自动执行预定义轨迹”均不算。

| ID | 简称 | 与本文关系 | 采集自动化 | 实时反馈 | target imperfection | Introduction 的主要用途 |
|---|---|---|---|---|---|---|
| P01 | Niola 2011 3-D scanner | Peripheral | 实验台人工/规定运动 | 否 | 否 | 线激光三角测量背景 |
| P02 | Yin 2014 sphere TCF | Supporting | 半自动规定运动 | 否 | 否 | 球约束和旋转/平移激励 |
| P03 | Li 2021 reconstruction | Supporting | 设计轨迹后自动执行 | 否 | 仅列误差源 | 窄轮廓限制、运动几何与重建精度 |
| P04 | Zhang 2013 on-machine | Supporting | NC 预定义自动执行 | 否 | 否 | “自动执行”边界样例 |
| P05 | Zou 2021 GAN | Direct competitor | 示教轨迹后自动采集 | 否 | 否 | 人工标注/多样性/数据量瓶颈 |
| P06 | Yang 2022 path planning | Direct competitor | 模型规划 | 否 | 否 | 自动路径、未知手眼启动困难 |
| P07 | Chen 2018 disk | Predecessor | 半自动规定扫掠 | 否 | 否 | 简单二维标定件 |
| P08 | Chen 2026 four-plane gauge | Direct competitor | 未提出自动采集 | 否 | 仿真制造误差 | 单帧二维轮廓恢复 6-DoF 的专用几何路线 |
| P09 | Antonello 2017 automatic HEC | Direct competitor | 两阶段闭环/受控采集 | 是（相机） | 否 | chicken-and-egg 与反馈启动 |
| P10 | Xiao 2022 automatic welding | Direct competitor | 闭环视觉反馈采集 | 是（相机图案） | 否 | 未知手眼+窄视场的闭环解决方案 |
| P11 | Wang 2020 simultaneous | Direct competitor | 人工改位姿，联合求解 | 否 | 否 | 相机可见棋盘格的同步三标定 |
| P12 | Zou 2020 DRL | Direct competitor | 示教轨迹后自动执行 | 否 | 否 | 端到端误差传播与采集边界 |
| P13 | Paschke 2022 optimized views | Direct competitor | 离线模型优化位姿 | 否 | 否 | 观测设计、粗手眼依赖 |
| P14 | Hu 2013 line laser | Supporting | 预定义自动执行 | 否 | 否 | 联合求解与无需精密物体 |
| P15 | Hu 2011 pointer | Supporting | 预定义自动执行 | 否 | 否 | 平面约束旁系谱系 |
| P16 | Heikkilä 2014 procedures | Supporting | 人工五位姿 | 否 | 否 | 简单标定件与工业调试 |
| P17 | Lavest 1998 inaccurate pattern | Predecessor | 人工多视图 | 否 | 完整静态几何 | RQ2 的共享几何和尺度 gauge 起点 |
| P18 | Choi 2016 two lidars | Supporting | 人工多次观测 | 否 | 仅容忍非正交 | 二维信息不足与退化配置 |
| P19 | Fernández-Moral 2D LRFs | Supporting | 人工场景观测 | 否 | 否 | 二维特征稀缺、粗初值与可观性 |
| P20 | Peters 2020 ICP | Supporting | 规定旋转/离线 ICP | 否 | 隐式场景 | 宽视场 2-D lidar 与窄 profiler 的边界 |
| P21 | Wu 2021 global HEC | Peripheral | 不讨论采集 | 否 | 否 | 全局求解器背景 |
| P22 | Liška 2018 sphere | Predecessor | 粗标定后规定扫描 | 否 | 否 | 数百轮廓与精密球代价 |
| P23 | Xu 2022 straight edges | Direct predecessor | 半自动 | 否 | 否 | 普通直边、人工运动范围、未来自动化 |
| P24 | Zhong 2025 single plane | Direct competitor | 随机/人工采集 | 否 | 未建模；明确承认影响 | 单平面线结构光与普通平面精度冲突 |
| P25 | Xie 2015 sphere accuracy | Predecessor | 规定方向扫描 | 否 | 否 | 扫描区域/运动方向影响精度 |
| P26 | Hagemann 2022 deformation | Predecessor | 人工多图 | 否 | 静态+动态形变 | 共享静态误差、逐帧动态误差、独立验证 |
| P27 | Strobl 2008 dimensions | Predecessor | 人工多视图 | 否 | 全局尺寸误差 | 手眼中共享目标尺寸联合估计 |
| P28 | Strobl 2011 imperfect plane | Predecessor | 人工多视图 | 否 | 完整静态非理想平面 | 与 RQ2 最接近的概念先驱 |
| P29 | Horn 2021 online ego-motion | Supporting | 在线估计，不控运动 | 否 | N/A | 在线标定仍需激励和可观性 |
| P30 | Xing 2023 arbitrary objects | Supporting | 随机采集/离线 ICP | 否 | 隐式完整场景 | 非平行转轴、重叠与初值限制 |
| P31 | Albarelli 2010 inaccurate target | Predecessor | 人工多图 | 否 | 完整静态几何 | 内部残差不等于外部精度 |
| P32 | Sharifzadeh 2020 single plane | Direct competitor | 四条人工启动+自动圆形扫描 | 否 | 否；用高平面度板 | 单平面 profiler 的启动依赖 |
| P33 | SCALAR 2019 | Direct predecessor | 人工大量位姿 | 否 | 只承认平面度 | 平面位姿联合估计≠形貌估计 |
| P34 | Pavlovčič 2021 simultaneous | Direct competitor | 人工初装+15预定义位姿 | 否 | 否；精密 3-D 件 | 周期自动标定与专用高精度件 |
| P35 | Carlson 2015 planes | Predecessor | 人工多平面 | 否 | 否 | 粗初值、非平行平面与收敛 |
| P36 | Gautam 2025 multiple profilers | Direct competitor | 六个预定义姿态 | 否 | 滤波处理粗糙/平面度 | “首个自动单平面”冲突文献 |
| P37 | Yin 2013 robot self-calibration | Supporting | 人工视觉对点 | 否 | 否 | 位姿数量/分布和人工负担 |

### 1.1 优先级

- 第一优先级（写 Introduction 必须精读/正面讨论）：P06、P08–P10、P13、P17、P23–P28、P31–P36。
- 第二优先级（共同构成证据链）：P03、P05、P07、P11–P12、P14、P16、P18–P20、P22、P25、P29–P30、P37。
- Peripheral：P01、P02、P04、P15、P21。它们已完成审阅，但不应占据 Introduction 的主要篇幅。

## 2. 按主题重组

### A. 2-D profiler hand-eye calibration and geometric targets

- 球：P02、P22、P25。优点是共同球心提供强三维约束；代价是已知高精度半径、扫掠规划和大量轮廓。
- 圆盘/多平面/专用三维件：P07、P08、P16、P18–P19、P34–P35。它们通过更丰富几何弥补二维截面的信息缺失，但目标制造、布置或可见性要求更高。
- 直边/任意物体：P03、P20、P23、P30。它们降低专用标定件依赖，却把困难转移到同一特征保持、点云重叠、初值或大量观测。
- 单平面：P24、P32–P33、P35–P36。它们最接近现场易部署方向，也是 RQ1 启动和 RQ2 模型失配最集中的文献组。

### B. Calibration artefact simplification

谱系可写为：精密三维件/球（P02、P22、P25、P34）→ 平面圆盘（P07）→ 多平面约束但未知位姿（P18、P35）→ 普通直边（P23）→ 单平面（P24、P32、P33、P36）。

必须同时写出代价转移：目标越简单，单次观测提供的约束越少，采集姿态、初值、运动多样性和模型正确性就越关键。P24 尤其重要，因为作者明确承认普通加工平面使球拟合/重投影表现劣于传统精密标定件。

### C. Automatic / autonomous calibration and data acquisition

- 预定义执行：P04、P05、P12、P14、P34、P36。
- 模型驱动规划：P06、P13、P32 的后半阶段。
- 真正闭环反馈采集：P09、P10；两者都利用相机中可识别的完整标定图案，而不是 profiler-only 二维轮廓。
- 仍明显依赖人工初始采集：P22、P23、P32、P34。

因此我们安全的缺口不是“以前没有自动标定”，而是：**已核实的闭环启动方法依赖相机图案；已核实的二维 profiler 单平面方法仍依赖粗手眼、人工启动扫描或预定义模型运动。**

### D. Motion diversity / pose selection / degeneracy

- 方程可观性和退化：P18、P19、P30、P32、P33、P35。
- 面向精度的规定运动：P03、P22、P25、P36。
- 离线优化视角：P13、P32。
- 在线/闭环保持有效观测：P09、P10。
- 只强调数据多样但仍靠示教：P05、P12。

这组文献支持两个不同论点，不能混写：一是“姿态需要非退化”；二是“在运动过程中观测还必须持续存在”。前者不自动解决后者。

### E. Imperfect calibration targets and scene-structure estimation

- 完整共享目标几何：P17。
- 共享全局尺寸：P27。
- 共享静态非理想平面：P28、P31。
- 共享静态误差 + 逐帧动态形变：P26。
- 只估计理想平面位姿：P24、P33、P35；这不属于 target-imperfection modeling。
- 仅滤波或选择高精度板：P32、P36；这也不属于联合形貌估计。

### F. Single-plane 2-D-profiler methods

- P32：一个高平面度单平面；四条人工扫描和粗手眼启动，随后自动执行 48 个圆形扫描位姿。
- P24：普通加工单平面；允许随机测量并用模拟退火挽救初值，但明确承认外部球/重投影精度较差。
- P33：平面位姿与手眼/机器人运动学联合估计；板面仍是理想平面，0.05 mm 平面度只是误差源。
- P35：多个非平行理想平面和粗手眼初值。
- P36：六个预定义平面姿态完成多 profiler 标定；粗糙度/平面度只经平均滤波和直线拟合处理。

### G. Direct competitors

RQ1 的直接竞争者为 P06、P09、P10、P13、P23、P32、P34、P36；RQ2 的直接概念先驱为 P17、P26–P28、P31，而 profiler 单平面模型的直接对照为 P24、P32–P33、P35–P36。P08 是“增加专用几何，使单轮廓直接恢复 6-DoF”的替代路线，必须讨论其精度和部署代价，但它不是同一问题设定。

## 3. 文献演进链

### 3.1 RQ1：从“能求解”到“未知手眼下仍能自主取得有效数据”

```text
精密球/三维标定件提供强几何约束
P02, P16, P22, P25, P34
        ↓ 降低标定物制造与现场布置负担
圆盘、多平面、直边、单平面
P07, P18, P23, P24, P32, P35, P36
        ↓ 简单几何使采集姿态和可观性更关键
规定轴向/圆形/星形运动与离线优化位姿
P03, P13, P22, P25, P32, P36
        ↓ 但生成目标机器人位姿仍依赖粗手眼或已知目标关系
相机图案下的两阶段启动与闭环采集
P09, P10（P06 为模型规划而非闭环）
        ↓ 尚未由已下载文献解决的限定缺口
profiler-only 二维轮廓 + 未知准确手眼 + 在线保持指定双边有效
        ↓ 我们的 RQ1
双边角点反馈、自举局部参考、局部视觉伺服、姿态多样性检查
```

严谨写法应突出“输入模态和闭环机制的组合缺口”，而不能把自动化本身宣称为新问题。

### 3.2 RQ2：从“精确目标”到“跨位姿共享的固定形貌”

```text
精确已知 target geometry
传统球/棋盘格/平面方法
        ↓ 发现名义几何误差会偏置标定
未知完整目标点坐标联合估计
P17
        ↓ 用低维共享量减少自由度
未知网格尺度/纵横尺寸
P27
        ↓ 允许大范围非理想平面
共享静态三维 target geometry
P28, P31
        ↓ 区分固定制造误差与每次搬动的形变
共享静态误差 + 每帧动态变形
P26
        ↓ 2-D profiler 单平面文献仍通常使用理想平面
P24, P32, P33, P35, P36
        ↓ 我们的 RQ2
在平板局部坐标系中估计低维 surface morphology，所有机器人位姿共享同一组参数
```

RQ2 的文献新意应定位为：**把相机标定中“共享静态目标几何”的思想迁移并约束到 profiler-only 单平面手眼问题，而不是声称首次考虑不准确标定物。**

## 4. Introduction 的论点—证据索引

以下每条均可直接发展成 Introduction 句子。括号内 ID 指向完整 CSV 中的短引文、页码和证据等级。

### 4.1 Background 与二维信息限制

1. **二维轮廓仪单次只给出一个截面内的距离/高度数据，缺少相机图案或三维点云中的完整空间特征。** 证据：P03、P08、P18、P19、P23、P24、P32、P35。
2. **因此，二维 profiler 手眼标定通常依赖多位姿运动或额外几何约束。** 证据：P07、P18、P22–P25、P32–P36。
3. **宽视场 2-D lidar 的场景 ICP 不能不加条件地迁移到窄视场 profiler。** 证据：P03、P20、P30。

### 4.2 标定物简化的动机与代价

4. **专用三维件和高精度球能提供强约束，但会增加制造、布置和操作成本。** 证据：P03、P07、P11、P16、P22–P24、P32、P34。
5. **二维 profiler 标定物已沿圆盘、直边和单平面方向简化。** 证据：P07、P23、P24、P32、P33、P35、P36。
6. **标定几何越简单，采集运动的可观性和多样性越关键。** 证据：P13、P18–P19、P23–P25、P30、P32、P35–P36。

### 4.3 自动化、启动与闭环采集

7. **人工示教/人工采集是现场标定的重要时间和一致性瓶颈。** 证据：P05、P06、P09、P10、P12、P13、P23、P32、P34、P37。
8. **已有“automatic”方法包含至少三种不同机制：预定义执行、模型规划和实时反馈闭环，不能混称。** 证据：P04–P06、P09–P10、P12–P14、P32、P34、P36。
9. **未知手眼会造成 bootstrap difficulty：系统不知道怎样移动才能维持目标可见并获得有效观测。** 证据：P06、P09、P10、P13、P19、P22、P32、P35。
10. **相机图案系统已经存在闭环启动方案，因此我们的区别必须限定为 profiler-only 二维轮廓反馈。** 证据：P09、P10；对照：P06、P11、P14。
11. **二维 profiler 单平面方法仍常依赖粗手眼、人工初始扫描或离线模型。** 证据：P13、P22–P24、P32、P35–P36。

### 4.4 姿态质量而非单纯数量

12. **更多数据不自动等于更准；退化运动、偏置分布或测量几何差的观测可能不提供有效信息。** 证据：P03、P06、P13、P18–P19、P25、P29–P30、P32、P35–P37。
13. **现有工作已研究离线观测优化、条件数和可观性，因此“使用 NBV/选择好位姿”本身不能作为充分创新。** 证据：P13、P18、P25、P29–P30、P32、P36；另应直接加入 Yang–Rebello–Waslander 2023（见待补文献）。
14. **“方程非退化”与“运动过程中目标始终可见”是两个不同问题。** 非退化证据：P18、P30、P32、P35；可见性证据：P09、P10、P13、P23。

### 4.5 不准确标定物与共享形貌

15. **真实 target geometry 的系统性偏差能够污染相机/手眼参数。** 证据：P17、P26–P28、P31；profiler 侧的现象证据：P24、P33、P36。
16. **固定制造误差应在所有观测之间共享，而不是为每个位姿各自自由拟合。** 证据：P17、P26–P28、P31。
17. **动态弯曲与静态制造形貌是不同误差类别，需要不同共享结构。** 证据：P26。
18. **估计理想平面的位姿不等于估计其非理想形貌。** 反例：P24、P33、P35；对照：P28、P31。
19. **滤波或使用高平面度板不能被表述为 target geometry joint estimation。** 反例：P32、P36；对照：P17、P26–P28、P31。
20. **内部 surface/reprojection residual 降低不足以证明手眼外参更准，必须使用独立任务/球体等外部验证。** 证据：P26、P28、P31；profiler 直接警示：P24。

## 5. 可直接采用的 Introduction 逻辑骨架

1. 先限定传感器输入：工业二维 profiler 每次只产生截面轮廓，无法像相机一样直接观察完整棋盘图案（P03、P08、P24、P32）。
2. 回顾几何标定物路线：精密球/三维件精度强但部署重，研究逐步转向圆盘、直边和单平面（P07、P22–P25、P32–P36）。
3. 引出 RQ1：简单几何把困难转移到数据采集；现有 profiler 方法依赖人工范围、粗手眼或离线位姿，而相机闭环方案依赖可检测图案（P09、P10、P13、P23、P32）。
4. 明确 RQ1 的限定缺口：准确手眼未知、只有实时二维轮廓时，如何自主保持指定几何观测并形成多样运动。
5. 再引出独立的 RQ2：单平面方法几乎都把目标当作理想平面；但普通加工平面的误差已被 profiler 文献观察到会降低外部精度（P24、P33、P36）。
6. 接相机领域谱系：共享目标几何、共享尺寸和静态/动态形变建模已证明“目标误差与传感器参数必须分离”（P17、P26–P28、P31）。
7. 明确 RQ2 的限定缺口：这些思想尚未在已审阅的 profiler-only 单平面手眼框架中，以低维、跨机器人位姿共享的 surface morphology 形式出现。
8. 最后再陈述本文两个相对独立但共同服务现场部署的模块；不要用“NBV 首创”或“单平面首创”作为贡献标题。

## 6. Do-not-claim list

| 目前不能安全写的表述 | 冲突证据 | 建议改写 |
|---|---|---|
| “现有单平面方法都不能自动标定。” | P14、P24、P32、P36 | “现有自动化程度不同；profiler-only 方法多为预定义执行、随机采集或依赖粗初值。” |
| “本文首次使用单平面进行二维 profiler 手眼标定。” | P24、P32、P33、P35、P36 | “本文面向普通非理想单平面上的闭环采集与共享形貌估计。” |
| “本文首次实现全自动手眼标定。” | P09、P10、P14、P36；Tsai–Lenz 1989 | 加上 `profiler-only、未知准确手眼、指定双边反馈` 等严格条件，并在检索扩展后再判断。 |
| “现有 automatic 方法都是预定义轨迹。” | P09、P10 | “多数 profiler 方法是预定义/模型规划；相机图案系统已有闭环反馈采集。” |
| “未知手眼下的自动采集从未被研究。” | P06、P09、P10、P32 | “该启动问题已在相机/粗扫描框架中研究，但二维 profiler-only 双边反馈仍缺少对应方案。” |
| “使用 NBV 是本文创新。” | P13；Yang–Rebello–Waslander 2023 | 把创新落在 `profile-only 可行观测模型/伺服` 或新的精度目标，而不是 NBV 名称。 |
| “随机采集一定需要很多数据，且精度一定较差。” | P24 认为随机测量可成功；P36 仅六姿态 | 改成“随机/未设计采集不保证在给定预算内获得良好条件与持续有效观测”。 |
| “所有已有方法都忽略标定物不准确。” | P17、P26–P28、P31 | 限定为“已审阅的 profiler 单平面手眼方法未显式联合估计跨位姿共享表面形貌”。 |
| “所有单平面 profiler 方法都完全忽略平面度。” | P32 选用 ±21 µm 板；P33 列平面度；P36 滤波 | 改成“它们未把非理想表面形貌作为跨位姿共享未知量联合估计”。 |
| “联合估计平面参数就是考虑了平面不平。” | P24、P33、P35 | 区分理想平面的法向/距离与空间变化的高度函数参数。 |
| “只要标定残差下降就证明外参精度提高。” | P26、P28、P31、P24 | 必须报告独立球重建、留出姿态或外部计量指标。 |
| “普通平板天然能达到高精度。” | P24 明确报告普通加工面导致较差球误差 | 写成待验证命题；精度贡献必须由 A/B/C 形貌消融和独立球实验支持。 |
| “二维 profiler 完全没有图像。” | P06、P10、P11、P24 使用含相机的线结构光系统；商业设备也可能提供强度/调试图像 | 精确写“本文算法仅使用二维度量轮廓，不依赖可识别图案的二维相机图像”。 |
| “闭环”只要运行过程中检查观测有效即可成立。 | P13、P32 为离线/预定义；P09、P10 才是观测决定运动 | 写清反馈量是否实际改变下一步运动；仅安全检查应称 online validation，而非 closed-loop acquisition。 |

## 7. Recommended papers to retrieve

这些论文尚不在本子目录中，不能作为上面主表中的已核实直接证据。部分在 `/workspace/docs` 其他位置已有副本，建议复制到本证据库并单独完成页码审计。

| 论文 | 年份 | 被哪篇已下载论文引用/关联 | 为什么值得补齐 |
|---|---:|---|---|
| R. Y. Tsai and R. K. Lenz, “A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration” | 1989 | P09、P13、P32 等 | 自动星形采集的经典起点；防止错误宣称“自动手眼首创”。 |
| Jun Yang, Jason Rebello, Steven L. Waslander, “Next-Best-View Selection for Robot Eye-in-Hand Calibration” | 2023 | 与 P13 的视角优化主题直接相关；本地在 `/workspace/docs/Yang2023_NextBestView_HandEye.pdf` | 直接竞争 NBV；明确“坏观测可降低精度”和信息增益公式。 |
| W. R. Scott, G. Roth, J.-F. Rivest, “View Planning for Automated Three-Dimensional Object Reconstruction and Inspection” | 2003 | P13 引用 [17] | 自动视角规划的综述谱系。 |
| W. R. Scott, “Model-Based View Planning” | 2009 | P13 引用 [18] | 区分离线模型规划与在线视觉闭环。 |
| M. R. Driels and U. S. Pathre, “Significance of Observation Strategy on the Design of Robot Calibration Experiments” | 1990 | P13 引用 [24] | 观测策略影响标定条件与精度的经典依据。 |
| Y. Sun and J. M. Hollerbach, “Observability Index Selection for Robot Calibration” | 2008 | P13 引用 [26] | 可观性指标与位姿选择，不应把这一概念误认作本文首创。 |
| G. Q. Wei and G. Hirzinger, “Active Self-Calibration of Hand-Mounted Laser Range Finders” | 1998 | P32 引用 [26] | 标题直接涉及 active laser range finder self-calibration，必须核实其传感器输入和运动闭环。 |
| N. Andreff, R. Horaud and B. Espiau, “On-Line Hand-Eye Calibration” | 1999 | P03 引用 [8] | 在线求解与在线采集的边界；避免把二者混称。 |
| Ulrich & Hillemann, “Uncertainty-Aware Hand–Eye Calibration” | 2024 | 本地在 `/workspace/docs/Uncertainty-Aware_HandEye_Calibration.pdf` | 可用于区分机器人位姿不确定度与本文 target morphology；不是 RQ2 的同一误差源。 |
| Lei Huang, Qican Zhang and Anand Asundi, “Flexible Camera Calibration Using Not-Measured Imperfect Target” | 2013 | P26 引用 [8] | 属于静态不精确目标谱系，可补强 P17/P28/P31 之间的发展链。 |
| Annika Hagemann, Moritz Knorr, Holger Janssen and Christoph Stiller, “Inferring Bias and Uncertainty in Camera Calibration” | 2021 | P26 引用 [6] | 给出独立误差/不确定度评估思路，可支撑“训练残差不足以评价形貌模型”。 |

## 8. 当前证据能支持的最窄研究缺口

综合 37 篇已下载论文，当前最安全、最有辨识度的缺口表述是：

> 现有工作已分别实现了相机图案下的闭环自动手眼采集、二维 profiler 的简单几何/单平面标定以及不准确相机标定板的共享几何估计；然而，在准确手眼外参尚不可用、算法输入仅为二维度量轮廓的条件下，如何利用实时轮廓反馈自主维持指定简单几何观测并形成姿态多样数据，同时把同一普通平板的固定非理想形貌作为跨机器人位姿共享未知量从手眼外参中分离，尚未在本次审阅的文献中得到统一解决。

这句话仍应写成“在本次系统审阅范围内未发现”，而不是绝对的“首次”。若用于正式投稿，还需对第 7 节中 Tsai–Lenz、Wei–Hirzinger、Yang 2023 等关键前驱完成数据库检索和全文页码核验。
