#!/usr/bin/env python3
"""Draw a paper-ready flowchart of the real automatic star-seed acquisition.

The diagram is intentionally generated from a frozen real run rather than from
an imagined sequence.  Run-specific evidence (branch order, accepted frame
counts and observed rollback branches) is read from ``seeds.json`` and
``seed_collection.log``.  The generic state-transition structure follows the
current seed-collection implementation.

No ROS process is started and no production file is modified.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RUN = Path(
    "/workspace/data/calibration_runs/"
    "20260820_144753_焊接书_位置1_真机_1"
)
DEFAULT_OUTPUT = Path("/workspace/star_seed_acquisition_flowchart")


@dataclass(frozen=True)
class RunEvidence:
    branch_labels: tuple[str, ...]
    inlier_counts: tuple[int, ...]
    observation_count: int
    rollback_branches: tuple[str, ...]
    preflight_mode: str
    preflight_executed: bool


BRANCH_DISPLAY = {
    "reference": "reference",
    "ry_positive": "R_y(+) ",
    "ry_negative": "R_y(-)",
    "rx_positive": "R_x(+)",
    "rx_negative": "R_x(-)",
    "rx_ry_positive": "R_xR_y(+)",
}


def read_evidence(run_dir: Path) -> RunEvidence:
    seeds_path = run_dir / "seeds.json"
    log_path = run_dir / "seed_collection.log"
    if not seeds_path.is_file():
        raise FileNotFoundError(f"missing seed data: {seeds_path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"missing seed log: {log_path}")

    payload = json.loads(seeds_path.read_text(encoding="utf-8"))
    seeds = payload.get("seeds", [])
    labels = tuple(str(seed.get("label", "unknown")) for seed in seeds)
    inliers = tuple(len(seed.get("frames", [])) for seed in seeds)
    dynamic = payload.get("dynamic_preflight", {})

    current_branch = ""
    rollback_branches: list[str] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        target_match = re.search(r"\]: target ([A-Za-z0-9_+-]+)\s*$", line)
        if target_match:
            current_branch = target_match.group(1)
        if "rollback, next rotation step" in line and current_branch:
            if current_branch not in rollback_branches:
                rollback_branches.append(current_branch)

    return RunEvidence(
        branch_labels=labels,
        inlier_counts=inliers,
        observation_count=int(payload.get("observation_count", sum(inliers))),
        rollback_branches=tuple(rollback_branches),
        preflight_mode=str(dynamic.get("mode", "unknown")),
        preflight_executed=bool(dynamic.get("executed", False)),
    )


def q(value: str) -> str:
    """Quote a DOT string."""
    return json.dumps(value, ensure_ascii=False)


def table_label(title: str, lines: list[str], title_color: str = "#17365D") -> str:
    rows = [
        f'<TR><TD ALIGN="CENTER"><FONT COLOR="{title_color}"><B>{html.escape(title)}</B></FONT></TD></TR>'
    ]
    rows.extend(
        f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="10">{line}</FONT></TD></TR>'
        for line in lines
    )
    return "<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"1\" CELLPADDING=\"1\">" + "".join(rows) + "</TABLE>>"


def node(name: str, label: str, **attrs: str) -> str:
    rendered = [f"label={label}"]
    rendered.extend(f"{key}={q(value)}" for key, value in attrs.items())
    return f"{name} [{', '.join(rendered)}];"


def edge(source: str, target: str, label: str = "", **attrs: str) -> str:
    rendered: list[str] = []
    if label:
        rendered.append(f"label={q(label)}")
    rendered.extend(f"{key}={q(value)}" for key, value in attrs.items())
    suffix = f" [{', '.join(rendered)}]" if rendered else ""
    return f"{source} -> {target}{suffix};"


def display_branch(label: str) -> str:
    return BRANCH_DISPLAY.get(label, label)


def build_dot(evidence: RunEvidence, language: str) -> str:
    is_zh = language == "zh"
    font = "Noto Sans CJK SC" if is_zh else "DejaVu Sans"

    if is_zh:
        title = "基于双边轮廓反馈的六种子星型自动采集流程"
        subtitle = "1 个参考位姿 + 5 个自适应旋转分支（NBV 之前）"
        stage1, stage2, stage3 = "I  引导初始化", "II  自适应星型分支循环", "III  数据输出与后续接口"
        txt = {
            "start": ("开始", ["人工介入仅发生在初始对准"]),
            "manual": ("人工设置参考观察", ["将目标板面轮廓移入紫色 ROI", "并靠近期望 80 mm 参考线"]),
            "align": ("初始双边观测稳定？", ["双断点可见、机器人静止", "且初始工作包络通过"]),
            "detect": ("锁定同一物理角的双断点", ["ROI 内断点检测与有序身份分配", "初始化两个 CSRT 局部跟踪器"]),
            "refcap": ("定点采集参考种子", ["目标 18 帧；MAD 剔除端点离群帧", "至少保留 14 个同步内点"]),
            "refok": ("参考批次有效？", ["安全双边 + 同步位姿 + 足够内点"]),
            "track": ("进入种子跟踪模式", ["保持 endpoint_u / endpoint_v 身份", "记录最后一个已验证安全位姿"]),
            "probe": ("一次性三轴局部平移探测", ["ΔS=[Δs_x  Δs_y  Δs_z]", "ΔQ=[Δq_x  Δq_y  Δq_z]", "J⁰=ΔS·ΔQ⁺ ∈ ℝ²ˣ³"]),
            "branches": ("建立星型分支队列", ["名义顺序：R_y(+), R_y(-), R_x(+),", "R_x(-), R_xR_y(+)", "必要时使用非平行备用分支"]),
            "return": ("返回已验证参考位姿", ["分段关节回程 + 有界模板重捕获", "重置本分支角度、步长和前馈模型"]),
            "select": ("选择下一旋转分支", ["局部法兰轴；目标角 10°"]),
            "rotate": ("执行一个姿态微步", ["初始 Δθ=1°；验证后可加速到 2°", "叠加已学习的局部平移前馈"]),
            "measure": ("停稳并读取同步观测", ["CSRT 跟踪双断点；提取 s=[x_mid,L]ᵀ", "L 为两个指定物理断点的间距"]),
            "hard": ("仍为目标双边且处于硬安全域？", ["观测安全、身份连续", "50 mm ≤ L ≤ 120 mm"]),
            "rollback": ("保护性恢复", ["退回最后有效位姿；减小 Δθ", "丢弃不可信前馈并重新跟踪", "重复失败：回参考位姿/换备用分支"]),
            "update": ("更新实测局部模型", ["更新旋转引起的特征漂移率", "累计已完成旋转角"]),
            "soft": ("双特征进入伺服目标带？", ["|x_mid| ≤ 10 mm", "70 mm ≤ L ≤ 90 mm"]),
            "servo": ("闭环局部平移修正", ["e=[e_x,e_L]ᵀ", "Δq_F=−kJᵀ(JJᵀ+λ²I)⁻¹e", "有效实测运动用于 Broyden 更新 J"]),
            "angle": ("达到该分支目标姿态？", ["累计旋转角 ≥ 10°"]),
            "capture": ("定点采集该分支种子", ["18 帧同步观测；端点 MAD 筛选", "同时检查旋转多样性"]),
            "capok": ("≥14 个内点且姿态多样？", []),
            "reject": ("拒绝本分支数据", ["返回参考位姿并尝试备用分支", "必要时以安全的部分姿态作为备选"]),
            "accept": ("接受一个物理种子位姿", ["保存每个内点帧的 T_BF、轮廓和双端点"]),
            "all": ("已获得 6 个物理种子？", ["reference + 5 个有效分支"]),
            "save": ("写入 seeds.json", [f"本次：6 个物理位姿，{evidence.observation_count} 个同步内点帧", "停止种子阶段的双断点跟踪"]),
            "handoff": ("六种子联合初解", ["共享模型 12-DOF-V2 初始化", "随后才进入 NBV（本图范围之外）"]),
        }
        yes, no = "是", "否"
        evidence_title = "本次真机运行证据"
        formula_title = "双特征与死区误差"
        optional = f"动态 ±X/±Y 预检为可选工程模块；本次 mode={evidence.preflight_mode}，未执行。"
    else:
        title = "Closed-loop automatic acquisition of six star-shaped seed poses"
        subtitle = "One stationary reference + five adaptive rotation branches (before NBV)"
        stage1, stage2, stage3 = "I  Guided initialization", "II  Adaptive star-branch loop", "III  Output and downstream handoff"
        txt = {
            "start": ("Start", ["Manual intervention is confined to initial alignment"]),
            "manual": ("Set the reference observation", ["Move the target-surface chord into the guide ROI", "and close to the desired 80-mm reference line"]),
            "align": ("Stable valid dual-edge observation?", ["Both breakpoints visible; robot stationary", "and initial working envelope satisfied"]),
            "detect": ("Lock the two physical breakpoints", ["ROI breakpoint detection and ordered assignment", "Initialize two local CSRT trackers"]),
            "refcap": ("Capture the stationary reference seed", ["Target 18 frames; endpoint MAD rejection", "At least 14 synchronized inlier frames"]),
            "refok": ("Reference batch valid?", ["Safe pair, synchronized pose, sufficient inliers"]),
            "track": ("Enter seed-tracking mode", ["Preserve endpoint_u / endpoint_v identity", "Store the last verified safe pose"]),
            "probe": ("One-time 3-axis translation probe", ["ΔS=[Δs_x  Δs_y  Δs_z]", "ΔQ=[Δq_x  Δq_y  Δq_z]", "J⁰=ΔS·ΔQ⁺ ∈ ℝ²ˣ³"]),
            "branches": ("Initialize the star-branch queue", ["Nominal order: R_y(+), R_y(-), R_x(+),", "R_x(-), R_xR_y(+)", "Non-parallel fallback branches if required"]),
            "return": ("Return to the verified reference", ["Staged joint return + bounded-template reacquisition", "Reset branch angle, step and feed-forward model"]),
            "select": ("Select the next rotation branch", ["Local flange axis; 10° target"]),
            "rotate": ("Execute one orientation micro-step", ["Start at Δθ=1°; accelerate to 2° after verification", "Add learned local-translation feed-forward"]),
            "measure": ("Settle and read synchronized feedback", ["Track both breakpoints; extract s=[x_mid,L]ᵀ", "L is the intended physical-breakpoint separation"]),
            "hard": ("Intended edge pair inside hard guard?", ["Safe observation and continuous identities", "50 mm ≤ L ≤ 120 mm"]),
            "rollback": ("Protective recovery", ["Rollback to last valid pose; reduce Δθ", "Discard unsafe feed-forward and reacquire", "Repeated failure: reference recovery / fallback branch"]),
            "update": ("Update measured local models", ["Update rotation-induced feature-drift rate", "Accumulate achieved rotation"]),
            "soft": ("Dual feature inside servo target band?", ["|x_mid| ≤ 10 mm", "70 mm ≤ L ≤ 90 mm"]),
            "servo": ("Closed-loop local translation", ["e=[e_x,e_L]ᵀ", "Δq_F=−kJᵀ(JJᵀ+λ²I)⁻¹e", "Accepted measured motions update J by Broyden"]),
            "angle": ("Branch target reached?", ["Accumulated rotation ≥ 10°"]),
            "capture": ("Capture the stationary branch seed", ["18 synchronized frames; endpoint MAD filtering", "Check rotational diversity"]),
            "capok": ("≥14 inliers and diverse pose?", []),
            "reject": ("Reject this branch sample", ["Return to reference and try a fallback branch", "A safe partial orientation may be retained as fallback"]),
            "accept": ("Accept one physical seed pose", ["Store T_BF, surface profile and endpoints per inlier"]),
            "all": ("Six physical seeds collected?", ["Reference + five valid branches"]),
            "save": ("Write seeds.json", [f"This run: 6 physical poses, {evidence.observation_count} synchronized inliers", "Stop seed-stage breakpoint tracking"]),
            "handoff": ("Joint six-seed initialization", ["Shared 12-DOF-V2 initialization", "NBV starts afterward (outside this figure)"]),
        }
        yes, no = "Yes", "No"
        evidence_title = "Evidence from the selected real run"
        formula_title = "Dual feature and dead-band error"
        optional = f"Measured ±X/±Y preflight is optional; this run used mode={evidence.preflight_mode} and skipped it."

    def label(key: str) -> str:
        heading, lines = txt[key]
        return table_label(heading, lines)

    branch_text = " → ".join(display_branch(x) for x in evidence.branch_labels)
    inlier_text = ", ".join(str(x) for x in evidence.inlier_counts)
    rollback_text = ", ".join(display_branch(x) for x in evidence.rollback_branches) or ("无" if is_zh else "none")
    evidence_lines = (
        [
            f"实际顺序：{html.escape(branch_text)}",
            f"各位姿内点帧：[{html.escape(inlier_text)}]；合计 {evidence.observation_count}",
            f"发生回退的分支：{html.escape(rollback_text)}",
        ]
        if is_zh
        else [
            f"Observed order: {html.escape(branch_text)}",
            f"Inlier frames per pose: [{html.escape(inlier_text)}]; total {evidence.observation_count}",
            f"Branches with rollback: {html.escape(rollback_text)}",
        ]
    )
    formula_lines = [
        "s=[x<SUB>mid</SUB>, L]<SUP>T</SUP>, &nbsp; s<SUP>*</SUP>=[0,80 mm]<SUP>T</SUP>",
        "e<SUB>x</SUB>=0 if |x<SUB>mid</SUB>|≤10 mm; otherwise e<SUB>x</SUB>=x<SUB>mid</SUB>",
        "e<SUB>L</SUB>=0 if 70≤L≤90 mm; otherwise e<SUB>L</SUB>=L−80 mm",
    ]

    lines = [
        "digraph star_seed_acquisition {",
        f"graph [rankdir=TB, splines=polyline, overlap=false, nodesep=0.20, ranksep=0.34, pad=0.18, margin=0.04, bgcolor=\"white\", fontname={q(font)}, fontsize=17, labelloc=\"t\", label={q(title + chr(10) + subtitle)}, compound=true, newrank=true];",
        f"node [shape=box, style=\"rounded,filled\", fillcolor=\"#F7FAFC\", color=\"#4A6075\", penwidth=1.25, fontname={q(font)}, fontsize=10, margin=\"0.12,0.07\"] ;",
        f"edge [color=\"#43576A\", penwidth=1.25, arrowsize=0.72, fontname={q(font)}, fontsize=9, fontcolor=\"#334E68\"] ;",
        "subgraph cluster_init {",
        f"label={q(stage1)}; color=\"#90B8D8\"; penwidth=1.5; style=\"rounded\"; bgcolor=\"#F5FAFE\"; fontname={q(font)}; fontsize=13;",
        node("start", label("start"), shape="oval", fillcolor="#DCEEFF", color="#2F6F9F"),
        node("manual", label("manual"), shape="parallelogram", fillcolor="#FFF4D6", color="#B88214"),
        node("align", label("align"), shape="diamond", fillcolor="#FFF9E8", color="#B88214", margin="0.08,0.04"),
        node("detect", label("detect"), fillcolor="#E8F3FB", color="#3976A8"),
        node("refcap", label("refcap"), fillcolor="#E9F7EF", color="#2E7D57"),
        node("refok", label("refok"), shape="diamond", fillcolor="#F0F8F4", color="#2E7D57", margin="0.08,0.04"),
        node("track", label("track"), fillcolor="#E8F3FB", color="#3976A8"),
        node("probe", label("probe"), fillcolor="#EDE9FA", color="#6554A3"),
        node("preflight_note", table_label("Optional preflight" if not is_zh else "可选动态预检", [html.escape(optional)], "#7A5A00"), shape="note", style="filled,dashed", fillcolor="#FFFBEA", color="#C49A2C"),
        "}",
        "subgraph cluster_loop {",
        f"label={q(stage2)}; color=\"#A7C9A8\"; penwidth=1.5; style=\"rounded\"; bgcolor=\"#F8FCF8\"; fontname={q(font)}; fontsize=13;",
        node("branches", label("branches"), fillcolor="#EDF6ED", color="#4D8A50"),
        node("return", label("return"), fillcolor="#F1F6F9", color="#607D8B"),
        node("select", label("select"), fillcolor="#EDF6ED", color="#4D8A50"),
        node("rotate", label("rotate"), fillcolor="#EDE9FA", color="#6554A3"),
        node("measure", label("measure"), fillcolor="#E8F3FB", color="#3976A8"),
        node("hard", label("hard"), shape="diamond", fillcolor="#FFF4ED", color="#B85C32", margin="0.08,0.04"),
        node("rollback", label("rollback"), fillcolor="#FDEDEC", color="#B5483B", style="rounded,filled,dashed"),
        node("update", label("update"), fillcolor="#EDE9FA", color="#6554A3"),
        node("soft", label("soft"), shape="diamond", fillcolor="#FFF9E8", color="#B88214", margin="0.08,0.04"),
        node("servo", label("servo"), fillcolor="#FFF4D6", color="#B88214"),
        node("angle", label("angle"), shape="diamond", fillcolor="#F0F8F4", color="#2E7D57", margin="0.08,0.04"),
        node("capture", label("capture"), fillcolor="#E9F7EF", color="#2E7D57"),
        node("capok", label("capok"), shape="diamond", fillcolor="#F0F8F4", color="#2E7D57", margin="0.08,0.04"),
        node("reject", label("reject"), fillcolor="#FDEDEC", color="#B5483B", style="rounded,filled,dashed"),
        node("accept", label("accept"), fillcolor="#DFF3E8", color="#2E7D57"),
        node("all", label("all"), shape="diamond", fillcolor="#EAF4F8", color="#3976A8", margin="0.08,0.04"),
        node("formula", table_label(formula_title, formula_lines, "#4D3C86"), shape="note", fillcolor="#F5F2FD", color="#7B68B3"),
        "}",
        "subgraph cluster_out {",
        f"label={q(stage3)}; color=\"#A9B7C4\"; penwidth=1.5; style=\"rounded\"; bgcolor=\"#FAFBFC\"; fontname={q(font)}; fontsize=13;",
        node("save", label("save"), shape="cylinder", fillcolor="#E8F3FB", color="#3976A8"),
        node("handoff", label("handoff"), shape="oval", fillcolor="#EDE9FA", color="#6554A3"),
        node("evidence", table_label(evidence_title, evidence_lines, "#17365D"), shape="note", fillcolor="#F7FAFC", color="#7890A4"),
        "}",
        edge("start", "manual"),
        edge("manual", "align"),
        edge("align", "detect", yes, color="#2E7D57", fontcolor="#2E7D57"),
        edge("align", "manual", no + ("：继续调整" if is_zh else ": readjust"), color="#B5483B", fontcolor="#B5483B"),
        edge("detect", "refcap"),
        edge("refcap", "refok"),
        edge("refok", "track", yes, color="#2E7D57", fontcolor="#2E7D57"),
        edge("refok", "manual", no + ("：重新对准/采集" if is_zh else ": realign/recapture"), color="#B5483B", fontcolor="#B5483B"),
        edge("refok", "preflight_note", "", style="dashed", arrowhead="none", color="#C49A2C"),
        edge("track", "probe"),
        edge("probe", "branches"),
        edge("branches", "return"),
        edge("return", "select"),
        edge("select", "rotate"),
        edge("rotate", "measure"),
        edge("measure", "hard"),
        edge("hard", "update", yes, color="#2E7D57", fontcolor="#2E7D57"),
        edge("hard", "rollback", no, color="#B5483B", fontcolor="#B5483B"),
        edge("rollback", "rotate", ("恢复成功" if is_zh else "Recovered"), color="#B5483B", fontcolor="#B5483B"),
        edge("rollback", "reject", ("重复失败" if is_zh else "Repeated failure"), style="dashed", color="#B5483B", fontcolor="#B5483B"),
        edge("update", "soft"),
        edge("soft", "angle", yes, color="#2E7D57", fontcolor="#2E7D57"),
        edge("soft", "servo", no, color="#B88214", fontcolor="#9B6E0B"),
        edge("servo", "measure", ("停稳并复测" if is_zh else "Settle and remeasure"), color="#B88214", fontcolor="#9B6E0B"),
        edge("soft", "formula", "", style="dashed", arrowhead="none", color="#7B68B3"),
        edge("angle", "rotate", no, color="#3976A8", fontcolor="#3976A8"),
        edge("angle", "capture", yes, color="#2E7D57", fontcolor="#2E7D57"),
        edge("capture", "capok"),
        edge("capok", "accept", yes, color="#2E7D57", fontcolor="#2E7D57"),
        edge("capok", "reject", no, color="#B5483B", fontcolor="#B5483B"),
        edge("reject", "return", color="#B5483B"),
        edge("accept", "all"),
        edge("all", "return", no + ("：下一分支" if is_zh else ": next branch"), color="#3976A8", fontcolor="#3976A8"),
        edge("all", "save", yes, color="#2E7D57", fontcolor="#2E7D57"),
        edge("save", "handoff"),
        edge("save", "evidence", "", style="dashed", arrowhead="none", color="#7890A4"),
        # Keep side notes to the right without forcing them into the main path.
        "{rank=same; refok; preflight_note;}",
        "{rank=same; hard; rollback;}",
        "{rank=same; soft; formula;}",
        "{rank=same; save; evidence;}",
        "}",
    ]
    return "\n".join(lines) + "\n"


def render(dot_path: Path, stem: Path) -> list[Path]:
    dot = shutil.which("dot")
    if dot is None:
        raise RuntimeError("Graphviz 'dot' was not found on PATH")
    outputs: list[Path] = []
    for extension, extra in (("svg", []), ("pdf", []), ("png", ["-Gdpi=300"])):
        destination = stem.with_suffix(f".{extension}")
        command = [dot, *extra, f"-T{extension}", str(dot_path), "-o", str(destination)]
        subprocess.run(command, check=True)
        outputs.append(destination)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--language",
        choices=("zh", "en", "both"),
        default="both",
        help="Generate Chinese, English, or both versions (default: both).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = read_evidence(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    languages = ("zh", "en") if args.language == "both" else (args.language,)
    generated: list[Path] = []
    for language in languages:
        stem = args.output_dir / f"star_seed_acquisition_flowchart_{language}"
        dot_path = stem.with_suffix(".dot")
        dot_path.write_text(build_dot(evidence, language), encoding="utf-8")
        generated.append(dot_path)
        generated.extend(render(dot_path, stem))
    print("Generated:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
