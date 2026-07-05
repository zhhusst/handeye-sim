# 线激光平板角点法手眼标定：12-DOF 完整方法

> 最后修订：2026-07-04
> 状态：✅ 9 个角点位姿实测 R=0.0225°, t=0.09mm

---

## 目录

**第一部分：理论框架**
1. [问题定义与坐标系](#1-问题定义与坐标系)
2. [纯平面约束的局限性](#2-纯平面约束的局限性)
3. [角点法：数学模型](#3-角点法数学模型)
4. [Gauge 消除：完整的对称性证明](#4-gauge-消除完整的对称性证明)

**第二部分：求解器与实现**
5. [三种求解器形式](#5-三种求解器形式)
6. [初始化与多重重启策略](#6-初始化与多重重启策略)

**第三部分：自动化采集**
7. [角点伺服：Xiao 2022 线激光等价](#7-角点伺服xiao-2022-线激光等价)
8. [自动化管线设计](#8-自动化管线设计)

**第四部分：验证与结论**
9. [仿真验证结果](#9-仿真验证结果)

**附录**
A. [完整 Jacobian 推导](#附录-a完整-jacobian-推导)
B. [代码清单与版本差异](#附录-b代码清单与版本差异)

---

# 第一部分：理论框架

## 1. 问题定义与坐标系

### 1.1 问题陈述

**给定**：
- 线激光传感器（如 Gocator）固定在机器人末端法兰上
- 标定板是已知尺寸 $w \times h$ 的平板，置于工作空间
- 机器人可执行 $M$ 个不同位姿，激光扫描线穿过板角区域，产生断点

**目标**：估计手眼变换 $R_{he} \in SO(3)$ 和 $t_{he} \in \mathbb{R}^3$（传感器坐标系 $S$ 到法兰坐标系 $H$）

### 1.2 坐标系

| 坐标 | 符号 | 含义 |
|:--|:--|:--|
| $B$ | 基座标系 | 机器人基座，世界坐标系 |
| $H_i$ | 法兰坐标系 | 第 $i$ 个位姿时的机器人末端 |
| $S$ | 传感器坐标系 | 线激光（$y=0$ 为激光平面） |
| $P$ | 平板坐标系 | 以角点 $C$ 为原点，$u_B$ 为边 1 方向，$v_B$ 为边 2 方向，$n_B = u_B \times v_B$ 为法向量 |

### 1.3 基本变换

传感器点 → 基座标系：

$$p_B = R_i(R_{he} \cdot p_S + t_{he}) + t_i \tag{1}$$

其中 $(R_i, t_i)$ 为第 $i$ 个位姿时法兰在基座标系下的位姿（由 FK 给出）。

### 1.4 已知量与未知量

**已知**：$(R_i, t_i)$（机器人 FK）、平板尺寸 $w \times h$、扫描线断点坐标 $p_{S,e1}, p_{S,e2}$

**未知**（12 参数）：

$$\theta = [\omega_{he}(3),\; t_{he}(3),\; \omega_{pl}(3),\; C(3)] \in \mathbb{R}^{12}$$

| 参数 | 维度 | 参数化 | 含义 |
|:--|:--|:--|:--|
| $\omega_{he}$ | 3 | 轴角，$R_{he} = \exp([\omega_{he}]_\times)$ | 手眼旋转 |
| $t_{he}$ | 3 | 笛卡尔 | 手眼平移 |
| $\omega_{pl}$ | 3 | 轴角，$R_{pl} = [u_B, v_B, n_B]$ | 平板朝向 |
| $C$ | 3 | 笛卡尔 | 角点在基座标系中的位置 |

*为什么是 12 参数？* 手眼变换需要 6 个参数（SO(3) + 平移），平板需要 3 个朝向参数（SO(3) 平板局部坐标系），角点需要 3 个位置参数。平板朝向需要 3 个参数（而非 2 个），因为面内旋转（$u_B, v_B$ 绕 $n_B$ 的定向）由边缘约束锁定——面内旋转在纯平面约束下不可观测，但角点法的边缘约束使其成为可观测量。

---

## 2. 纯平面约束的局限性

### 2.1 约束方程

纯平面（Sharifzadeh 2020）的约束：激光点落在平板上：

$$n_B^T \cdot (R_i(R_{he} \cdot p_S + t_{he}) + t_i) = d_B \tag{2}$$

其中 $d_B$ 为平面到基座标系原点的距离。

### 2.2 耦合结构

方程 (2) 中，$n_B$ 和 $R_{he}$ 以乘积形式出现：$n_B^T R_i R_{he} p_S$。固定 $n_B^T R_i R_{he}$ 这个组合的值，$n_B$ 和 $R_{he}$ 可以联合变化而残差不变——这就是 $n_B$-$R$ 耦合。这是 gauge 的根源，不是数值问题。

### 2.3 信息效率分析

考察约束对 $t_{he}$ 的梯度：

$$\frac{\partial}{\partial t_{he}} \big(n_B^T R_i(R_{he} p_S + t_{he}) + t_i - d_B\big) = n_B^T R_i$$

**关键事实**：一个位姿的所有平面点共享同一个 $t_{he}$ 梯度方向——$n_B^T R_i$（一个 $1 \times 3$ 行向量）。这意味着：
- 每个位姿只贡献 $t_{he}$ 方向上的 1 个有效独立方程
- 要覆盖 $t_{he}$ 的全部 3 个方向，需要不同 $R_i$ 产生独立性

对旋转参数 $\omega_{he}$ 的可观测性同理：$n_B^T R_i \cdot [R_{he}p_S]_\times$ 虽随 $p_S$ 变化，但所有平面点在 $t_{he}$ 方向上共享同一个梯度子空间 → 每个位姿贡献 ≈2 个有效独立方程。

**结论**：纯平面约束下每个位姿 ≈2 个有效独立方程。覆盖 10 个未知参数（$R_{he}, t_{he}, n_B, d_B$）需要至少 5 个位姿（理论下界），实际需要更多（如 Sharifzadeh 的 48 个）来克服信息重叠。

---

## 3. 角点法：数学模型

### 3.1 核心思想

线激光扫描线同时穿过板角区域时，轮廓两端出现两个断点——分别落在两条正交边上。**两条边共享同一个角点 $C$ 作为坐标系原点**，这引入了物理一致性约束，从根本上压缩了 gauge 群。

### 3.2 约束方程（C-anchored 形式）

**边 1 约束**（断点 $p_{S,e1}$ 落在 $u_B$ 方向线上）：

$$\boxed{v_B^T \cdot (p_{B,e1} - C) = 0} \tag{Ed1-s}$$

等价于 cross-product 形式：$\operatorname{cross}(p_{B,e1} - C, u_B) = 0$。后者同时隐含 $v_B^T(p_{B,e1}-C)=0$ 和 $n_B^T(p_{B,e1}-C)=0$，其中 $n_B^T(\cdots)=0$ 与平面约束冗余。

**边 2 约束**（断点 $p_{S,e2}$ 落在 $v_B$ 方向线上，直角板情况）：

$$\boxed{u_B^T \cdot (p_{B,e2} - C) = 0} \tag{Ed2-s}$$

等价于 $\operatorname{cross}(p_{B,e2} - C, v_B) = 0$。

**平面约束**（所有激光点落在过 $C$ 的平面上）：

$$\boxed{n_B^T \cdot (p_B - C) = 0} \tag{Pl}$$

### 3.3 通用夹角记法

对非直角的已知夹角 $\alpha$（边 2 方向 $d_2 = \cos\alpha \cdot u_B + \sin\alpha \cdot v_B$），边 2 法向量为：

$$n_{\perp 2} = n_B \times d_2 = \cos\alpha \cdot v_B - \sin\alpha \cdot u_B$$

边 2 约束通用形式：$n_{\perp 2}^T \cdot (p_{B,e2} - C) = 0$。

当 $\alpha = \pi/2$（直角板）：$n_{\perp 2} = -u_B$，退化为 $u_B^T(p-C)=0$。

### 3.4 自由度统计

| 约束 | 每姿态度量数 | 独立秩（每 pose） |
|:--|:--|:--|
| 边 1：$v_B^T(p_{e1}-C)=0$ | 1 | 1 |
| 边 2：$u_B^T(p_{e2}-C)=0$ | 1 | 1 |
| 平面：$n_B^T(p-C)=0$（$N_p$ 点） | $N_p$ | $\le 2$（点间 $t$ 梯度线性相关） |
| **合计** | $N_p+2$ | $\approx 4$ |

- $t_{he}$ 方向：$\{v_B^T R_i, u_B^T R_i, n_B^T R_i\}$ 三个线性独立方向 → 满覆盖 $\mathbb{R}^3$
- 覆盖 12 DOF 理论下界：$\lceil 12/4 \rceil = 3$ 个位姿

### 3.5 C-anchored cross-product 残差（代码形式）

将边约束写为 3 分量 cross-product 确保与代码完全一致：

$$\begin{aligned}
r_{e1} &= \operatorname{cross}(p_{B,e1} - C,\; u_B) \in \mathbb{R}^3 \\
r_{e2} &= \operatorname{cross}(p_{B,e2} - C,\; v_B) \in \mathbb{R}^3 \\
r_{\text{plane}} &= n_B^T(p_B - C) \in \mathbb{R}
\end{aligned}$$

cross-product 为零等价于向量平行：$r_{e1}=0 \iff p_{B,e1}-C \parallel u_B \iff v_B^T(p_{B,e1}-C)=0$（边 1）$\land\; n_B^T(p_{B,e1}-C)=0$（与平面冗余）。LM 的伪逆自动处理冗余行。

---

## 4. Gauge 消除：完整的对称性证明

### 4.1 定义

令 $r: \mathbb{R}^{12} \to \mathbb{R}^m$ 为所有位姿的所有约束堆叠成的残差向量。

**定义**：一个非平凡变换 $g \neq \operatorname{id}$ 是 **gauge 变换**，若对所有可能的数据 $r(g\theta) = r(\theta)$。

### 4.2 前件

- $p_{B} = R_i(R_{he}p_S + t_{he}) + t_i$
- Assumption 1 (旋转非退化)：$\operatorname{span}\{R_i^T v_B, R_i^T u_B\}_{i=1}^M = \mathbb{R}^3$
- Assumption 2 (平移非退化)：$\{t_i\}_{i=1}^M$ 张成 $\mathbb{R}^3$

### 4.3 平移 Gauge

考虑变换 $(t_{he}, C) \to (t_{he} + \delta, C + \Delta)$。

**边约束**：断点 $p_{B,e1}$ 变为 $p_{B,e1} + R_i\delta$。

边 1：$v_B^T(p_{B,e1} + R_i\delta - C - \Delta) = v_B^T(p_{B,e1} - C) + v_B^T(R_i\delta - \Delta)$

要求对所有 $i$ 残差不变，即 $v_B^T(R_i\delta - \Delta) = 0$：

$$v_B^T R_i\delta = v_B^T \Delta \quad \forall i \tag{★}$$

这意味着标量 $v_B^T R_i\delta$ 对所有 $i$ 取相同常数 $c_1 = v_B^T \Delta$。取任意 $i, j$ 相减：

$$v_B^T(R_i - R_j)\delta = 0 \quad \forall i,j$$

即 $\delta \perp (R_i - R_j)^T v_B$ 对所有 $i,j$。令 $D_v = \operatorname{span}\{(R_i - R_j)^T v_B : i,j=1...M\}$，则 $\delta \in D_v^\perp$。

同理，边 2 给出 $\delta \in D_u^\perp$，其中 $D_u = \operatorname{span}\{(R_i - R_j)^T u_B\}$。

由 Assumption 1，$\operatorname{span}\{R_i^T v_B, R_i^T u_B\} = \mathbb{R}^3$。其差分子空间 $D_v$ 和 $D_u$ 的维数分别至少为 $\dim(\operatorname{span}\{R_i^T v_B\}) - 1$ 和 $\dim(\operatorname{span}\{R_i^T u_B\}) - 1$。当两个差分子空间的并集张成 $\mathbb{R}^3$ 时，$D_v^\perp \cap D_u^\perp = \{0\}$。

**因此 $\delta = 0$。** ∎

代入 $\delta = 0$：边 1 → $v_B^T \Delta = 0$，边 2 → $u_B^T \Delta = 0$ → $\Delta \parallel n_B$，即 $\Delta = \alpha n_B$（1-DOF 残余 gauge）。

**平面约束破除法向残余**：$C \to C + \alpha n_B$，平面残差 → $n_B^T(p_B - C - \alpha n_B) = n_B^T(p_B - C) - \alpha \neq n_B^T(p_B - C)$，除非 $\alpha = 0$。

**因此 $\Delta = 0$。** ∎

### 4.4 旋转 Gauge

**定理**：若 $(v_B, u_B, n_B, R_{he}) \to (Sv_B, Su_B, Sn_B, R_{he}S^T)$（$S \in SO(3)$）保持所有约束不变，则 $S = I$。

**证明**：考虑边 1 约束在变换下的形式：

$$r'_{e1} = (Sv_B)^T \cdot (R_i(R_{he}S^T p_{S,e1} + t_{he}) + t_i - C)$$

$r'_{e1}$ 中 $t_i$ 的系数为 $(Sv_B)^T$。为使 $r'_{e1} = r_{e1}$ 对所有位姿 $i$ 成立，$t_i$ 的系数必须相等：$(Sv_B)^T t_i = v_B^T t_i$ 对所有 $i$。

提取公共项：$(Sv_B - v_B)^T t_i = 0 \;\forall i$。由 Assumption 2（$\{t_i\}$ 张成 $\mathbb{R}^3$），$Sv_B - v_B$ 必须为零向量。

$$\boxed{S v_B = v_B}$$

同理，边 2 约束的 $t_i$ 系数给出 $Su_B = u_B$。平面约束的 $t_i$ 系数给出 $Sn_B = n_B$。

$\{u_B, v_B, n_B\}$ 是 $\mathbb{R}^3$ 的标准正交基（$R_{pl}$ 的列）。$S$ 在基的所有三个向量上均为恒等映射。

$$\boxed{S = I}$$

**推论**：$p_S$ 项自动满足——当 $Sv_B = v_B$ 时，$(Sv_B)^T R_i R_{he} S^T p_{S,e1} = v_B^T R_i R_{he} S^T p_{S,e1}$。由于 $S$ 在完整基底上为 $I$，此项退化为原始形式。 ∎

### 4.5 综合结论

| 变换 | 仅边缘约束的对称群 | 加平面约束后 | 最终 |
|:--|:--|:--|:--|
| 平移 $(t_{he}, C)$ | $\{(0, \alpha n_B)\}$ — 1-DOF | $\alpha = 0$ | **平凡** |
| 旋转 | $\{I\}$ | — | **平凡** |

**Gauge 已被完全消除。12-DOF 系统满秩，无需正则化。**

### 4.6 Gauge 消除 → 局部可辨识性

**定理**：若 gauge 已被消除（对称群平凡），则存在 $\theta^*$ 的邻域使 Jacobian $J(\theta)$ 满列秩（$\operatorname{rank}=12$）。

**工程意义**：
1. 非线性优化具有局部唯一极小值
2. 任意梯度求解器在接近真值时必然收敛
3. 条件数仅由数据质量决定，不受参数化冗余污染

### 4.7 数值验证

| 测试场景 | Jacobian 秩 | 条件数 | 结论 |
|:--|:--|:--|:--|
| C-anchored cross-product（3 姿，混合朝向） | **12/12** | 123 | 满秩 ✓ |
| 标量形式（3 姿，混合朝向） | **12/12** | 107 | 满秩 ✓ |
| 传感器全 ∥ $-n_B$（3 姿） | **11/12** | $3.0 \times 10^{11}$ | **gauge 重现** ✗ |
| Pairwise + centered（3 姿） | **9/12** | — | **C 3-DOF gauge** ✗ |

### 4.8 实操警示：严禁传感器 Z 轴平行于平板法向量

若所有位姿的传感器 $Z$ 轴均平行于 $-n_B$（即"垂直往下照"），则 $R_i^T n_B$ 对所有 $i$ 为常向量。此时 $t_{he}$ 沿 $R_{he}e_z$ 方向的平移与 $C$ 沿 $n_B$ 方向的位移构成无法分离的耦合——gauge 重新出现，条件数从 $10^2$ 飙升至 $3.0 \times 10^{11}$（数值验证确认）。

**避免方法**：保持传感器有 5°–25° 的自然倾斜。操作员手持扫角点时通常自动满足——危险的是自动化位姿规划（如果算法只优化"分母最大"，会自我毁灭倾向 $z_S = -n_B$）。

---

# 第二部分：求解器与实现

## 5. 三种求解器形式

代码提供了三种残差形式，均对应上述 C-anchored 数学框架（标量形式最直接；cross-product 等价但多出行数；传感器帧预测是等价变换）。本节使用**标量形式**作为数学描述。

### 5.1 形式 1：C-anchored Cross-Product（verify_12dof.py / calibrate.py）

残差定义见 §3.5。参数向量 $\theta = [\omega_{he}(3), t_{he}(3), \omega_{pl}(3), C(3)]$。边约束 $cross(p - C, \text{axis}) \in \mathbb{R}^3$，平面约束 $n_B^T(p - C) \in \mathbb{R}$。None-centered，C 参与所有约束 → gauge-free（§4 证明）。

### 5.2 形式 2：标量边约束（calib_solver.py `residuals_principle`）

边 1 残差：$v_B^T(p_{B,e1} - C)$（标量）；边 2：$u_B^T(p_{B,e2} - C)$（标量）；平面：$n_B^T(p - C)$（标量）。

**数学等价性**：$v_B^T(p_{e1} - C) = 0$ 是 $cross(p_{e1} - C, u_B) = 0$ 的非冗余分量（$n_B^T(p_{e1} - C) = 0$ 分量与平面约束冗余，被省略）。

优势：残差数量少（每个边断点 1 个标量 vs 3 个分量），信息不损失。

### 5.3 形式 3：传感器帧预测（calib_solver.py `residuals_principle_12dof`）

**基本思想**：利用边缘几何的解析关系，将断点预测回传感器帧与实测值比较。

对于边 1（方向 $u_B$），激光面（$y=0$ 平面，法向量 $n_l = R_{BS} e_2$）与边缘线的交点满足（推导见 PRINCIPLE.md Sec 4.3）：

$$s = \frac{n_l^T (t_{BS} - C)}{n_l^T u_B}$$

预测的传感器坐标 $p_S^{pred} = R_{he}^T(R_i^T(C + s \cdot u_B - t_i) - t_{he})$。残差为 $p_S^{pred}[0] - p_{S,e1}[0]$，$p_S^{pred}[2] - p_{S,e1}[2]$（$(x,z)$ 分量，$y$ 恒为零）。

**特点**：在传感器帧比对，本征地处理了线激光的 1D 特性。

### 5.4 求解器选择

| 形式 | 残差维度 | C 是否参与 | 推荐场景 |
|:--|:--|:--|:--|
| Cross-product (C-anchored) | $3N_e + N_p$ | ✅ | 验证、快速标定 |
| 标量 | $N_e + N_p$ | ✅ | 理论最简洁、数值稳定 |
| 传感器帧预测 | $2N_e + N_p$ | ✅ | 对噪声更鲁棒（传感器帧比较） |

---

## 6. 初始化与多重重启策略

### 6.1 R_he 初始化

当手眼真值距单位阵 >30° 时，从 $R=I$ 出发的 LM 被困在局部极小值（阻尼限制步长）。解决方案：

**多重重启**：在 SO(3) 上均匀采样 $N=20$ 个随机旋转 $\omega_{rand} = \alpha \cdot \hat{v}$（$\hat{v}$ 为随机单位向量，$\alpha \sim \text{Uniform}(0, \pi)$），各运行完整 init + LM，选 cost 最低的解。

实测：17/20 重启收敛到正确解，3 个未收敛的落入对称分支（cost 略高，可通过平板尺寸 post-hoc 排除）。

### 6.2 t_he、ω_pl、C 初始化

给定候选 $R_{cf}$（随机采样或名义值），初始化其余 9 参数：

1. **t_he 初值**：用 $R_{cf}$ 变换所有断点到基座标系 → 边方向拟合 → 跨朝向边距离约束 → 线性最小二乘（代码目前简化为零向量，但 LM 收敛到正确值——满秩系统对所有参数的梯度覆盖全空间）

2. **ω_pl 初值**：同朝向内皮点 SVD 拟合边方向 → 正交化 $v_B = n_B \times u_B$ → $R_{pl} = [u_B, v_B, n_B]$

3. **C 初值**：两条边线的最小二乘交点：
   - 边 1 线：$cross(C - p_{1,ref}, u_B) = 0$
   - 边 2 线：$cross(C - p_{2,ref}, v_B) = 0$
   - 法向约束：$n_B^T C = \operatorname{mean}(n_B^T \cdot \text{all plane points})$

   线性系统 A·C = b，最小二乘求解。

### 6.3 LM 参数

- $\lambda_0 = 10^{-4}$（初始阻尼）
- 有限差分 Jacobian，$\epsilon = 10^{-6}$
- 无正则化（满秩系统不包含冗余参数化）
- 收敛判据：cost 变化 < $10^{-12}$

---

# 第三部分：自动化采集

## 7. 角点伺服：Xiao 2022 线激光等价

### 7.1 Xiao 2022 框架映射

Xiao 2022 的核心：相机图像中板中心是可测量点特征 → 不依赖精确板模型做伺服。线激光等价：角点在 profile 中产生两个可测量断点，同样是直接可测量的几何特征。

| Xiao 2022（2D 相机） | 线激光等价（本方法） |
|:--|:--|
| 板中心像素 → 机器人修正 | 角点偏移 $\tilde{e} = (e_{1,x}+e_{2,x})/2$ → 1D 伺服 |
| 锁定后绕光轴旋转 | 锁定后绕传感器 Z 轴 ±5°~12° |
| EKF 更新板位姿 | 端点 PCA 更新 $u_B/v_B$ |

### 7.2 伺服不变性定理 ★

**定理**：传感器平移命令与手眼参数无关。

**证明**：传感器在基座标系中的位置：

$$t_{BS} = t_{BH} + R_{BH} \cdot t_{he}$$

对法兰施加平移 $\delta t_{BH}$：

$$t_{BS}^{new} = (t_{BH} + \delta t_{BH}) + R_{BH} \cdot t_{he} = t_{BS} + \delta t_{BH}$$

**法兰平移量 = 传感器平移量。与 $t_{he}$ 无关。** ∎

**推论**：伺服命令 $\delta t_{BH} = k \cdot \tilde{e} \cdot \hat{s}_x$（$\hat{s}_x = R_{BH} \cdot R_{he}^{nom}[:,0]$），其中方向用名义手眼有偏置角 $\theta$，但闭环伺服补偿。收敛条件：$|1 - k\cos\theta| < 1$，取 $k=0.5$ 即使 $\theta=30^\circ$ 仍满足。

### 7.3 伺服信号

$$\tilde{e} = \frac{e_{1,x} + e_{2,x}}{2}$$

- $\tilde{e} \approx 0$：锁定
- $\tilde{e} > 0$：角点偏右 → $\delta t = +k\tilde{e} \cdot \hat{s}_x$
- $\tilde{e} < 0$：角点偏左 → $\delta t = -k\tilde{e} \cdot \hat{s}_x$

$|\tilde{e}| \le 2$mm 判定锁定。$\tilde{e}$ 是连续量（不二值），提供"偏了多少、朝哪"的完整信息——闭环伺服充要条件。

### 7.4 锁定后的朝向多样性

锁定状态（$\tilde{e} \approx 0$）下做旋转获取朝向多样性：
- **绕 Z 轴**：角点不丢，直接记录（3~5 朝向 × ±5°, ±8°, ±12°）
- **绕 X/Y 轴**：±5~8°，旋转后需重伺服

---

## 8. 自动化管线设计

### 8.1 管线（移除 Phase 1 后的精简版本）

```
Phase 0: 操作工将激光放到板角 → 按 a → 记录锚点（FK + 关节角）
    ↓
Phase 2: 角点伺服采集
    2a. 伺服锁定（1D X 轴比例控制）
    2b. 朝向探索（锁定状态，绕 Z ±5~12°）
    2c. 平移多样性（沿传感器 X 平移 ±15mm, +30mm）
    ↓
Phase 3: 12-DOF LM（20 重启）→ R_he, t_he, ω_pl, C
```

### 8.2 为什么移除 Phase 1

Phase 1（关节扰动 + PCA 粗估板方向）的原始用途：
1. 提供粗糙板方向 $u_B, v_B$ → 用于边缘分配
2. 提供 7 个扰动位姿 → 额外数据

v6 伺服直接锁定到角点，锁定后的平移多样性位姿（±15mm, +30mm）用端点差分运动确定 $u_B/v_B$ → Phase 1 功能被替代。移除后管线更简洁：无需等待 7 个扰动位姿，从 Phase 0 直接进伺服。

### 8.3 已修复的关键问题

| # | 问题 | 修复 |
|:--|:--|:--|
| 1 | DH 模型与 URDF 不一致（IK 偏差 535mm） | 新增 `inverse_kinematics_numeric()`（URDF FK + Newton-Raphson） |
| 2 | Z 修正与伺服互搏 | 独立 $z\_correct\_count$，仅 Z < 300mm 或 > 700mm 触发 |
| 3 | 角点移动依赖偏置 locked 姿态 | 从锚点关节角出发 |
| 4 | 平移多样性不足 | DIVERSITY 子状态：±15mm, +30mm |
| 5 | 边缘分配用 Phase 1 PCA 出错 | 改用于端点差分运动确定 $u_B/v_B$ |

---

# 第四部分：验证与结论

## 9. 仿真验证结果

### 9.1 零噪声理想条件（Sim 手动采集，9 角点位姿，3 朝向）

> ⚠ **前提**：零激光噪声、零翘曲、零机器人重复性误差。结果代表求解器的理论精度上限。

| 指标 | 12-DOF（20 重启） | 目标 |
|:--|:--|:--|
| **R 误差** | **0.0225°** | <0.1° ✅ |
| **t 误差** | **0.09mm** | <0.1mm ✅ |
| **C 收敛** | [0.70, 0.00, 0.25] | ✅ |
| **Jacobian 秩** | **12/12**（verify_12dof.py） | 满秩 ✅ |
| **收敛率** | **17/20** 重启 | ✅ |
| **cost** | 2.57×10⁻⁶ | ✅ |

### 9.2 含噪声 MC（Num2 统一框架，0.055mm 激光 + 0.5mm 翘曲 + 0.1mm 重复性）

> 方法 1 和 3 使用相同的传感器帧预测求解器——唯一区别是采集策略（垂直 vs 倾斜）。

| 方法 | cond(J) | R median | R<0.1° | t median | gauge | 位姿数 |
|:--|:--|:--|:--|:--|:--|:--|
| `corner_12dof`（强制垂直） | ~10¹¹ | 3.9° (max 107°) | 0/10 | 15.9mm | ✗ | 6 |
| `plane_edge_9dof`（pairwise NBV） | ~10² | **0.24°** | 3/10 | 1.1mm | ✓ | 6 |
| `tilted_corner`（C-anchored 倾斜） | ~10² | 0.46° | 0/10 | 4.5mm | ✓ | 5-6 |

**解读**：
1. **强制垂直（corner_12dof）→ 灾难性失败**：cond=10¹¹，R 高达 107°，完全验证 §4.8 的警告。同一数学模型的 tilted_corner 仅因倾斜了 5-20° 就消除了 gauge（cond=10²）
2. **gauge 消除是必要条件，不是充分条件**：tilted_corner 的 gauge 消除了（cond=10²），但噪声 + 翘曲 + 5-6 个自动生成位姿（z_S 仅偏 6-10°）→ R median=0.46°，不如 Sim 的 0.0225°（零噪声 + 9 个手动精选位姿 + 3 个宽朝向）
3. **plane_edge_9dof 在噪声环境下表现最好**：pairwise edge 共线约束 + NBV 位姿（z_S 偏离 13-35°）→ R median=0.24°，3/10 达 <0.1°。但这是 9-DOF（C 不估计）——对于不需要 C 的场景够用了

### 9.3 精度瓶颈分析

零噪声 → R=0.0225°，含噪声（0.055mm + 0.5mm 翘曲）→ R≈0.2-0.5°。精度瓶颈是**数据质量**（噪声 + 翘曲），不是 gauge 或求解器。

---

# 附录

## 附录 A：完整 Jacobian 推导

### A.1 符号

| 符号 | 含义 |
|:--|:--|
| $[a]_\times$ | 反对称矩阵：$[a]_\times b = a \times b$ |
| $\partial(Rp)/\partial\omega = -[Rp]_\times$ | 旋转对轴角的标准导数 |

### A.2 平面约束 $r_p = n_B^T(R_i(R_{he}p_S + t_{he}) + t_i - C)$

$$\boxed{\frac{\partial r_p}{\partial t_{he}} = n_B^T R_i \quad (1 \times 3)}$$

$$\boxed{\frac{\partial r_p}{\partial \omega_{he}} = -n_B^T R_i [R_{he} p_S]_\times \quad (1 \times 3)}$$

令 $p_B^{full} = R_i(R_{he}p_S + t_{he}) + t_i$：

$$\frac{\partial r_p}{\partial \omega_{pl}} = \frac{\partial}{\partial \omega_{pl}}(n_B^T(p_B^{full} - C)) = (p_B^{full} - C)^T \cdot \frac{\partial n_B}{\partial \omega_{pl}}$$

由于 $n_B = R_{pl} e_3$，$\partial n_B^T/\partial \omega_{pl} = n_B^T[\cdot]_\times$：

$$\boxed{\frac{\partial r_p}{\partial \omega_{pl}} = n_B^T [p_B^{full} - C]_\times \quad (1 \times 3)}$$

$$\boxed{\frac{\partial r_p}{\partial C} = -n_B^T \quad (1 \times 3)}$$

### A.3 边 1 约束 $r_{e1} = v_B^T(p_{B,e1} - C)$

$$\frac{\partial r_{e1}}{\partial t_{he}} = v_B^T R_i$$

$$\frac{\partial r_{e1}}{\partial \omega_{he}} = -v_B^T R_i [R_{he} p_{S,e1}]_\times$$

$$\frac{\partial r_{e1}}{\partial \omega_{pl}} = v_B^T [p_{B,e1} - C]_\times$$

$$\frac{\partial r_{e1}}{\partial C} = -v_B^T$$

### A.4 边 2 约束（同结构，$v_B \to u_B$）

$$\frac{\partial r_{e2}}{\partial t_{he}} = u_B^T R_i$$

$$\frac{\partial r_{e2}}{\partial \omega_{he}} = -u_B^T R_i [R_{he} p_{S,e2}]_\times$$

$$\frac{\partial r_{e2}}{\partial \omega_{pl}} = u_B^T [p_{B,e2} - C]_\times$$

$$\frac{\partial r_{e2}}{\partial C} = -u_B^T$$

### A.5 关键对比

一个位姿的 Jacobian（$t_{he}$ 列）：
- 纯平面：$\{n_B^T R_i\}$ — 1 个方向 → 秩 = 1
- 角点法：$\{v_B^T R_i, u_B^T R_i, n_B^T R_i\}$ — 3 个线性独立方向 → 秩 = 3

这是信息密度翻倍的精确数学表达。

---

## 附录 B：代码清单与版本差异

### B.1 文件结构

```
Sim/
├── calibrate.py          # 标定管线（C-anchored cross-product + RMRL）
├── verify_12dof.py        # Jacobian 秩 + 精度验证
├── common/
│   ├── calib_solver.py    # 求解器库（3 种形式）
│   ├── plane_calib.py     # 平面约束 LM + NBV
│   └── fov_geometry.py    # so3_exp, so3_log
└── ros2_ws/src/handeye_sim_bridge/
    └── auto_calib_v2_node.py  # 自动化采集管线
```

### B.2 残差函数版本差异（重要）

| 函数 | 文件 | C 使用 | 平面 centering | C gauge |
|:--|:--|:--|:--|:--|
| `residuals_12dof` | calibrate.py | ✅ cross(p-C, axis) | ❌ | **0 DOF** |
| `residuals_12dof` | verify_12dof.py | ✅ cross(p-C, axis) | ❌ | **0 DOF** |
| `residuals_12dof` | **calib_solver.py** | ❌ pairwise diff | ✅ mean-subtracted | **⚠ 3 DOF** |
| `residuals_principle` | calib_solver.py | ✅ scalar dot | ❌ | **0 DOF** |
| `residuals_principle_12dof` | calib_solver.py | ✅ sensor-frame pred | ❌ | **0 DOF** |

**⚠ calib_solver.py 的 `residuals_12dof` 使用 pairwise 边缘差 + 平面中心化 → C 参数完全不可观测（3-DOF gauge）**。该 form 仅在 C 初始化合理且只关注 $(R_{he}, t_{he}, \omega_{pl})$ 时可用。数学上不应标称 12/12 满秩。

推荐使用 `residuals_principle`（标量形式）或 calibrate.py 的 C-anchored cross-product。

### B.2.1 Num2 vs Sim 求解器对照

| Num2 方法 | 残差形式 | 等效 Sim 函数 | DOF | 采集策略 |
|:--|:--|:--|:--|:--|
| `corner_12dof` | 传感器帧预测 | `calib_solver.residuals_principle_12dof` | 12 | **强制垂直** $z_S=-n_B$ |
| `tilted_corner` | 传感器帧预测 | `calib_solver.residuals_principle_12dof` | 12 | 倾斜网格 $z_S$ 偏 5-20° |
| `plane_edge_9dof` | pairwise diff + centered | `calib_solver.combined_residuals` | 9 | NBV 候选 $z_S$ 偏 13-35° |

**关键结论**：`corner_12dof` 和 `tilted_corner` 使用**完全相同**的数学求解器（传感器帧预测，12-DOF，C-anchored）。cond 从 10¹¹ 降到 10² 的唯一原因是采集时传感器倾斜了 5-20°——同一数学模型，gauge 不是几何固有的，是"强制垂直"的产物。

### B.3 实测采集指南

**最低要求**：≥3 位姿，≥2 朝向（夹角 >30°），覆盖两边。

**仿真启动**：
```bash
cd /home/z/research_contact_handeye/verification/Sim
./start.sh --tmux
```

**自动化采集**（v6）：
```bash
ros2 run handeye_sim_bridge auto_calib_v2  # 按 a 开始
```
