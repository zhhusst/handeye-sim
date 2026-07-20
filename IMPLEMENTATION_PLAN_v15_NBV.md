# v15 NBV 引导蠕动 — 实现方案

> 2026-07-08

## 目标

用几何 NBV 替代固定轴向蠕动，提高自动采集的朝向多样性，使 tilt_std > 5°，R_err < 0.1°。

## 可行性分析

| 可用的 (传感器数据场景) | 不可用的 (需要板模型) |
|:--|:--|
| tilt（传感器Z vs 世界Z，水平板假设） | 精确的 n_B, d_B |
| Profile 数据 → 近似平面法向 | u_B, v_B 方向 |
| e1/e2 角点断点 | 板尺寸 w_m, h_m |
| 当前传感器位姿 (TF + 名义手眼) | 角点 C 在世界坐标 |
| Z_center, span | FIM 所需的 d_B |

**结论**: Num1 FIM 和 Num2 候选生成都无法直接使用。采用几何 NBV（传感器局部坐标系的 pitch×yaw 网格 + 几何多样性评分）。

## 核心改动

只改 `_start_crawl()` 和新增 3 个方法。保留现有 servo/step/lock/diversity/Z伺服/span伺服 全流程不动。

```
原流程: _start_crawl() → 固定轴(T+绕cross(z_S,wZ)pre + Y+绕sensorY post)
         → _do_crawl_step(5°) → servo → lock → div
         → 累积 ≥ 10° → 录制3帧 → 换轴

新流程: _start_crawl() → NBV候选评估(pitch×yaw网格)
         → 几何多样性评分 → 选 top-2 目标朝向
         → 对每个目标: _do_crawl_toward(target_R)
            → 计算 geodesic 剩余角距离
            → 沿 geodesic 旋转 3-5° 朝目标
            → servo → lock
            → 重复直到到达目标 (剩余 < 3°)
            → 录制3帧 (±15mm 平移)
```

## 新增方法

### 1. `_generate_nbv_candidates()`

在传感器局部坐标系生成 pitch×yaw 网格候选朝向。

```python
def _generate_nbv_candidates(self):
    """生成传感器帧 pitch×yaw 网格候选朝向
    
    绕传感器 X (pitch) 和传感器 Y (yaw) 旋转，
    post-multiply 保证角点位移最小。
    过滤 tilt ∉ [15°, 50°] 的候选。
    
    Returns:
        list of dict: [{R_BH, pitch, yaw, tilt}, ...]
    """
    T = self._get_hand_pose()
    R_BH, t_BH = ros_tf_to_matrix(T)
    
    candidates = []
    for pitch in np.linspace(-20, 20, 5):    # 绕传感器 X: -20°..20°, 5 步
        for yaw in np.linspace(-20, 20, 4):  # 绕传感器 Y: -20°..20°, 4 步
            # 合成传感器帧旋转 (Rx 后 Ry)
            Rx = rodrigues(np.array([1., 0., 0.]), np.deg2rad(pitch))
            Ry = rodrigues(np.array([0., 1., 0.]), np.deg2rad(yaw))
            R_delta = Rx @ Ry
            
            # post-multiply: R_new = R_BH @ R_delta (角点位移小)
            R_new = R_BH @ R_delta
            
            # 检查 tilt
            z_S = R_new @ self.R_he_nom[:, 2]
            tilt = float(np.rad2deg(np.arccos(np.clip(
                abs(np.dot(z_S, [0., 0., 1.])), 0., 1.))))
            
            if tilt < 15.0 or tilt > 50.0:
                continue  # 角点检测不安全
            
            candidates.append({
                'R_BH': R_new,
                't_BH': t_BH,
                'pitch': pitch,
                'yaw': yaw,
                'tilt': tilt,
            })
    
    self.get_logger().info(
        f'  NBV候选: {len(candidates)} 个 (pitch×yaw网格, tilt∈[15,50]°)')
    return candidates
```

### 2. `_score_and_select_nbv(candidates)`

几何多样性贪心选择 top-2。

```python
def _score_and_select_nbv(self, candidates):
    """基于角距离的几何多样性选择 top-2 候选
    
    如果已有采集的朝向 (从 records 中提取), 
    综合考虑与已采集朝向的分离度。
    
    Returns:
        list of dict: 选中的候选 (≤2 个)
    """
    if len(candidates) == 0:
        self.get_logger().warn('  NBV候选池为空, 无法选择')
        return []
    
    if len(candidates) == 1:
        return candidates
    
    # 提取已采集朝向的 R_BH
    collected_Rs = []
    for rec in self.records:
        if 'R_BH' in rec:
            collected_Rs.append(rec['R_BH'])
    
    def ang_dist(Ra, Rb):
        """旋转角距离 (0°~180°)"""
        Rd = Ra.T @ Rb
        tr = np.clip((np.trace(Rd) - 1) / 2, -1., 1.)
        return float(np.rad2deg(np.arccos(tr)))
    
    # 如果已有采集数据, 先用与已采集的最小距离过滤
    if collected_Rs:
        # 选 5 个与已采集最不接近的候选
        scored = []
        for c in candidates:
            min_dist = min(ang_dist(c['R_BH'], Rc) for Rc in collected_Rs)
            scored.append((min_dist, c))
        scored.sort(key=lambda x: -x[0])
        pool = [c for _, c in scored[:5]]
    else:
        pool = candidates
    
    # 贪心选 top-2: 最大化两候选间角距离
    # 第1个: tilt 最大 (偏好大倾斜 → 多样性)
    first = max(pool, key=lambda c: c['tilt'])
    selected = [first]
    
    if len(pool) > 1:
        # 第2个: 与第1个角距离最大
        best, best_dist = None, -1
        for c in pool:
            if c is first:
                continue
            d = ang_dist(c['R_BH'], first['R_BH'])
            if d > best_dist:
                best_dist = d
                best = c
        if best and best_dist > 5.0:  # 至少 5° 分离
            selected.append(best)
    
    for i, c in enumerate(selected):
        self.get_logger().info(
            f'  NBV #{i+1}: pitch={c["pitch"]:.0f}° yaw={c["yaw"]:.0f}° '
            f'tilt={c["tilt"]:.1f}°')
    if len(selected) > 1:
        self.get_logger().info(
            f'    角距离: {ang_dist(selected[0]["R_BH"], selected[1]["R_BH"]):.0f}°')
    
    return selected
```

### 3. `_do_crawl_toward(target_R)`

沿 geodesic 旋转一步朝目标。

```python
def _do_crawl_toward(self, target_R):
    """沿 geodesic 朝目标旋转 3-5° 一步
    
    计算当前 R → target_R 的 geodesic 方向,
    旋转 min(剩余角度, 5°) 朝目标。
    
    Args:
        target_R: 目标法兰姿态 (3×3)
    """
    T = self._get_hand_pose()
    if T is None:
        return False
    R_cur, t_cur = ros_tf_to_matrix(T)
    
    # 相对旋转
    R_rel = R_cur.T @ target_R
    tr = np.clip((np.trace(R_rel) - 1) / 2, -1., 1.)
    angle_remaining = np.rad2deg(np.arccos(tr))
    
    if angle_remaining < 3.0:
        # 已到达目标
        self._crawl_safe_R = R_cur
        self._crawl_safe_t = t_cur
        self._crawl_nbv_reached = True
        return False
    
    # 沿 geodesic 方向旋转 5° (或剩余角度, 取小)
    step_deg = min(5.0, angle_remaining)
    axis = so3_log(R_rel)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return False
    axis = axis / axis_norm
    
    # post-multiply (传感器帧): R_new = R_cur @ R_delta
    R_step = rodrigues(axis, np.deg2rad(step_deg))
    R_new = R_cur @ R_step
    
    tilt = self._tilt_of_R(R_new)
    self.get_logger().info(
        f'    蠕动→目标: 剩余{angle_remaining:.1f}° 步{step_deg:.0f}° '
        f'tilt≈{tilt:.1f}°')
    
    self._move_to_pose(R_new, t_cur, f'蠕→NBV目标 {step_deg:.0f}°')
    self._crawl_nbv_target_R = target_R  # 记录目标用于下次调用
    return True
```

## 修改 `_start_crawl()`

```python
def _start_crawl(self):
    """Phase 2b: NBV 引导蠕动 — geo-NBV 选朝向 → 目标导向蠕动"""
    self.get_logger().info('\\n╔══ Phase 2b: NBV引导蠕动 ══╗')
    
    # ── NBV 候选评估 ──
    candidates = self._generate_nbv_candidates()
    self._crawl_nbv_targets = self._score_and_select_nbv(candidates)
    
    if len(self._crawl_nbv_targets) == 0:
        # Fallback: v14 固定轴
        self.get_logger().warn('  NBV无可用候选, 回退到固定轴')
        self._start_crawl_v14()  # 原 _start_crawl 逻辑
        return
    
    self._crawl_nbv_idx = 0
    self._crawl_orientation_count = 0
    self._crawl_target_orientations = len(self._crawl_nbv_targets)
    self._crawl_used_groups = set()
    self._auto_phase = 'CRAWL'
    
    # 记录 Z 伺服目标
    f = self._extract_features()
    if f and f['e1'] is not None and f['e2'] is not None:
        self._crawl_nominal_Z = (f['e1'][1] + f['e2'][1]) / 2
    else:
        self._crawl_nominal_Z = 0.35
    
    self._crawl_nbv_reached = False
    self._start_next_nbv_target()
```

注：原 `_start_crawl()` 重命名为 `_start_crawl_v14()` 作 fallback。

## 修改 `_on_crawl_done()` 的 STEP/SERVO 处理

在现有 `_on_crawl_done` 的锁定逻辑后，替代 `_do_crawl_step()` 的调用，改为：

```python
# 原: if accum >= target: ...  elif margin_min < 0.015: ...  else: self._do_crawl_step()
# 新: 用 _do_crawl_toward 替代 _do_crawl_step

if self._crawl_nbv_reached:
    # 已到达 NBV 目标 → 录制 + 换下一个目标
    self._record_pose(f)
    self._crawl_div_offsets = [0.015, -0.015]
    self._crawl_div_idx = 0
    offset = self._crawl_div_offsets[0]
    R_BH = self._crawl_safe_R
    s_x = R_BH @ self.R_he_nom[:, 0]
    self._move_to_pose(R_BH, self._crawl_safe_t + offset * s_x,
                       f'蠕多样性 {offset*1000:+.0f}mm')
    self._crawl_state = 'DIV'
    self._crawl_nbv_reached = False
elif margin_min < 0.015:
    # FOV margin 修正 (同 v14)
    ...
else:
    # 继续朝 NBV 目标蠕动
    if hasattr(self, '_crawl_nbv_target_R'):
        self._do_crawl_toward(self._crawl_nbv_target_R)
    else:
        self._do_crawl_step()  # fallback
```

## 改动汇总

| 位置 | 改动 | 行数 |
|:--|:--|:--|
| 新增 `_generate_nbv_candidates()` | 传感器帧 pitch×yaw 候选生成 | ~35 行 |
| 新增 `_score_and_select_nbv()` | 几何多样性贪心选择 | ~50 行 |
| 新增 `_do_crawl_toward(target_R)` | geodesic 目标导向单步旋转 | ~35 行 |
| 重写 `_start_crawl()` | NBV 评估 + fallback | ~25 行 |
| 修改 `_on_crawl_done()` | 锁定后调 `_do_crawl_toward` 替代 `_do_crawl_step` | ~15 行 |
| 保留所有现有方法 | servo/Z伺服/span伺服/div/margin保护 — 不动 | 0 行 |

**总计**: ~160 行新增/修改。核心 servo 管线完全不动。

## 预期效果

- tilt_std 从 ~4° → > 7°（NBV 选择 tilt 分散的朝向）
- 朝向数从 1 → 2（几何多样性确保两个朝向绕不同轴）
- 不再依赖固定轴向 → 适应不同初始 tilt

## 回退安全

- NBV 候选池为空 → 自动回退 v14 固定轴
- 目标导向蠕动失败 → 保留现有 margin 保护 + back 机制

---

## 待优化 (v16 — 2026-07-08)

**根因分析**: 蠕动只探索了 6 DOF 中 2 个旋转角。servo 把平移自由度消灭——ẽ→0 意味着 X/Y/Z 平移被拉回到同一状态。±15mm 偏置方向单一、幅度固定。

### 优化 1：Phase 2 后先降 tilt，再 NBV

**问题**: 初始 tilt=49° 时候选池里 tilt 都在 [47,50] 窄带，NBV 选两个的 tilt_std 到不了 5°。

**方案**:
```
Phase 2 servo 锁 ẽ→0 后:
  绕 sensor X 小步 post-multiply 旋转（不依赖 R_he）
  每步 tilt 降 3°，目标 25-30°
  re-servo → 进入 Phase 2b NBV
```

**效果**: 候选池 tilt 范围从 [47,50] → [15,50]，候选数 ~3 → ~15，NBV 真正有空间发挥作用。

### 优化 2：放松锁定条件

**问题**: ẽ ≤ 2mm 锁定 → 录。每次锁定时 X 平移同一状态，消灭了平移多样性。

**方案**: 
- 角点在 FOV 内（margin > 10mm）→ **先录一帧（不 servo）**
- 然后 servo 到 ẽ≈0 → 录第二帧
- 两帧之间天然的 ẽ 差异 = 平移信息

### 优化 3：多方向多幅度偏置

**问题**: 当前偏置只沿 sensor X ±15mm。

**方案**: 沿三个传感器轴各两个幅度：
```python
offsets = [
    (s_x, +10), (s_x, +25),   # X: 不同 ẽ
    (s_x, -10), (s_x, -25),
    (s_y, +15), (s_y, -15),   # Y: 不同 span
    (s_z, -20), (s_z, -40),   # Z: 不同深度
]
# 每朝向 8+1=9 帧
```

### 优先级

① 投入产出比最高——降 tilt 让整个 NBV 管线有空间，且只用 post-multiply 不碰 R_he。

### 核心理念修正 (2026-07-08)

**唯一硬约束：激光线切到两条边（角点可见）。其他所有约束都可以放松。**

| 当前约束 | 可否放松 | 放松后收益 |
|:--|:--|:--|
| ẽ ≤ 2mm 锁定 | ✅ 删除，角点可见即录 | X 平移多样性 |
| span ∈ [50,200]mm | ✅ 放至 [10,300] | Y/Z 平移多样性 |
| 每步固定 5° | ✅ 可变步长 | 加速/减速探索 |
| servo 到 ẽ→0 | ✅ 不做，录自然 ẽ | 每帧有不同平移状态 |
| 固定旋转轴 | ✅ NBV 自由选 | 探索更广 SO(3) |

**新原则**：在角点可见的前提下，每到一个新位姿（不管 ẽ 多大、span 多宽），直接录一帧。servo 只在角点丢失时用于拉回。
