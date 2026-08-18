#!/usr/bin/env bash
# =============================================================================
# 精密球独立验证 —— 一键评价脚本（评价方法2，RCIM2025 §8 口径）
#
# 串联：球数据(npz) → ROR过滤 → pose CSV → 逐组+合并球拟合 → 图15
#
# 用法：
#   ./scripts/eval_ball.sh --npz <球数据.npz> --result <calibration_result.json>
#   ./scripts/eval_ball.sh --npz <球数据.npz> --result <结果.json> --radius 10.001 --z-gate 25 --out <自定义输出>
#
# 参数：
#   --npz     球面采集数据（sphere_acquisition.npz，移动扫描离散步进采集）
#   --result  标定结果（calibration_result.json，含 handeye 变换）
#   --radius  参考球半径 mm（D20GZ 精密球 = 10.001；论文球 = 17.4605）
#   --z-gate  ROR 帧级深度门限 mm（默认 25，偏差超过即剔除该帧）
#   --out     输出目录（默认 <球数据目录>/rcim_eval）
#
# 输出：
#   <out>/validation/scanner1/pose1..7.csv   ROR 过滤后的基座系点云
#   <out>/metrics.txt                        逐组 + 合并评价数值
#   <out>/Fig15_sphere_validation.png        论文口径图（评价2 原生）
#   <out>/Fig15_beautiful.png                美化版图（验收排版）
#   <out>/eval_summary.txt                   一键汇总（供直接阅读）
# =============================================================================
set -eo pipefail

cd /workspace

# ---- 解析参数 ----
NPZ=""
RESULT=""
RADIUS=10.001
Z_GATE=25.0
OUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --npz) NPZ="$2"; shift 2 ;;
        --result) RESULT="$2"; shift 2 ;;
        --radius) RADIUS="$2"; shift 2 ;;
        --z-gate) Z_GATE="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 2 ;;
    esac
done

if [[ -z "$NPZ" || -z "$RESULT" ]]; then
    echo "用法: ./scripts/eval_ball.sh --npz <球数据.npz> --result <calibration_result.json> [--radius 10.001] [--z-gate 25] [--out <目录>]"
    exit 2
fi

if [[ ! -f "$NPZ" ]]; then
    echo "错误: 找不到球数据文件 $NPZ" >&2
    exit 1
fi
if [[ ! -f "$RESULT" ]]; then
    echo "错误: 找不到标定结果文件 $RESULT" >&2
    exit 1
fi

# 默认输出目录 = 球数据同级 rcim_eval
if [[ -z "$OUT" ]]; then
    OUT="$(dirname "$NPZ")/rcim_eval"
fi
mkdir -p "$OUT"

echo "======================================================================"
echo " 精密球独立验证（评价方法2 / RCIM2025 §8）"
echo "======================================================================"
echo " 球数据   : $NPZ"
echo " 标定结果 : $RESULT"
echo " 参考半径 : ${RADIUS} mm"
echo " ROR门限  : ${Z_GATE} mm"
echo " 输出目录 : $OUT"
echo "----------------------------------------------------------------------"

# ---- 第 1 步：ROR 过滤 + 转 pose CSV ----
echo "[1/3] ROR 过滤与基座系转换 ..."
python3 "/workspace/paper_test/评价方法2/npz_to_pose_csv_ror.py" \
    --npz "$NPZ" \
    --result "$RESULT" \
    --outdir "$OUT/validation" \
    --z-gate-mm "$Z_GATE"

# ---- 第 2 步：评价方法2（数值 + 论文口径图） ----
echo "[2/3] 运行评价方法2（逐组 + 合并球拟合）..."
python3 "/workspace/paper_test/评价方法2/sphere_validation_rcim_v2.py" \
    --root "$OUT/validation" \
    --out "$OUT" \
    --radius "$RADIUS"

# ---- 第 3 步：美化版 Fig15 ----
echo "[3/3] 生成美化版 Fig15 ..."
python3 "/workspace/paper_test/评价方法2/fig15_beautiful.py" \
    --root "$OUT/validation" \
    --out "$OUT/Fig15_beautiful.png" \
    --radius "$RADIUS"

# ---- 汇总 ----
if [[ -f "$OUT/metrics.txt" ]]; then
    {
        echo "# 精密球独立验证汇总（评价方法2）"
        echo ""
        echo "- 球数据: $NPZ"
        echo "- 标定结果: $RESULT"
        echo "- 参考半径: ${RADIUS} mm, ROR 门限: ${Z_GATE} mm"
        echo ""
        grep -E "Pose[0-9]|delta_r|r_fit|sigma|RMSE|center" "$OUT/metrics.txt" | head -40
    } > "$OUT/eval_summary.txt"
    echo "----------------------------------------------------------------------"
    echo "✅ 评价完成！产物："
    echo "   数值汇总 : $OUT/metrics.txt"
    echo "   论文口径图: $OUT/Fig15_sphere_validation.png"
    echo "   美化版图 : $OUT/Fig15_beautiful.png"
    echo "   一键汇总 : $OUT/eval_summary.txt"
    echo "----------------------------------------------------------------------"
    cat "$OUT/eval_summary.txt"
else
    echo "⚠️ 未找到 metrics.txt，请检查评价方法2是否运行成功。" >&2
    exit 1
fi
