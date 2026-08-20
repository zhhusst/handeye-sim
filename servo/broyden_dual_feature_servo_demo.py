
"""
Broyden 双特征无标定视觉伺服 Demo

视觉特征:
    s = [x_mid, L]^T
目标:
    x_mid* = 0 mm
    L*     = 80 mm

控制输入:
    u = [q_a, q_z]^T
其中:
    q_a : 当前主要用于修正 x_mid 的法兰局部平移方向
    q_z : 法兰局部 Z 轴平移

算法:
1. 用两个 1 mm probe 初始化 2x2 visual-motor Jacobian J
2. 每轮使用阻尼伪逆:
       du = -lambda * J^T (J J^T + mu I)^(-1) e
3. 执行后用 Broyden 更新:
       J+ = J + ((ds - J du) du^T)/(du^T du)
4. 对每个控制分量限幅，模拟真实机器人安全微动

注意:
- plant() 只是为了演示“未知、非线性、带耦合”的实机映射；
- 真机中 plant() 不存在，直接由“机器人运动 -> Gocator新测量”替代；
- Broyden 不需要知道真实手眼矩阵。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET = np.array([0.0, 80.0])   # [x_mid(mm), L(mm)]

# ---------------------------
# 仅用于 Demo 的“未知真实系统”
# ---------------------------
def plant(u):
    """
    u = [q_a, q_z]，单位 mm
    返回 [x_mid, L]，单位 mm

    特意加入:
    - 两个控制轴之间的耦合
    - 轻微非线性
    以模拟真实系统中“q_a 也影响 L，q_z 也影响 x_mid”的情况。
    """
    q_a, q_z = u

    x_mid = (
        15.0
        + 0.80 * q_a
        + 0.15 * q_z
        + 0.003 * q_a * q_z
        + 0.40 * np.sin(q_z / 8.0)
    )

    L = (
        120.0
        + 0.25 * q_a
        + 1.60 * q_z
        + 0.006 * q_a**2
        - 0.002 * q_z**2
    )

    return np.array([x_mid, L], dtype=float)


def initialize_jacobian_by_probe(u0, probe_mm=1.0):
    """有限差分 probe 初始化 J 的两列。"""
    s0 = plant(u0)
    J = np.zeros((2, 2), dtype=float)

    for j in range(2):
        u_probe = u0.copy()
        u_probe[j] += probe_mm
        s_probe = plant(u_probe)
        J[:, j] = (s_probe - s0) / probe_mm

    return J


def damped_pseudoinverse_step(J, error, gain=0.8, damping=0.05):
    """
    du = -lambda J^T (J J^T + mu I)^(-1) e
    """
    A = J @ J.T + damping * np.eye(2)
    return -gain * J.T @ np.linalg.solve(A, error)


def broyden_update(J, du, ds, eps=1e-9):
    """
    Good Broyden rank-1 update:
    J+ = J + ((ds - J du) du^T)/(du^T du)
    """
    denom = float(du @ du) + eps
    return J + np.outer(ds - J @ du, du) / denom


def run_demo(
    max_iterations=20,
    probe_mm=1.0,
    gain=0.8,
    damping=0.05,
    max_step_mm=5.0,
    x_tol_mm=0.10,
    L_tol_mm=0.20,
):
    u = np.zeros(2, dtype=float)
    s = plant(u)

    # 1) 两个小 probe 得到初始 Jacobian
    J = initialize_jacobian_by_probe(u, probe_mm)

    records = []

    for k in range(max_iterations + 1):
        error = s - TARGET

        records.append(
            {
                "iteration": k,
                "q_a_mm": u[0],
                "q_z_mm": u[1],
                "x_mid_mm": s[0],
                "L_mm": s[1],
                "x_error_mm": error[0],
                "L_error_mm": error[1],
                "J_x_qa": J[0, 0],
                "J_x_qz": J[0, 1],
                "J_L_qa": J[1, 0],
                "J_L_qz": J[1, 1],
            }
        )

        if abs(error[0]) <= x_tol_mm and abs(error[1]) <= L_tol_mm:
            break

        # 2) 同时求两个平移方向
        du = damped_pseudoinverse_step(
            J,
            error,
            gain=gain,
            damping=damping,
        )

        # 3) 模拟真机微步限幅
        du = np.clip(du, -max_step_mm, max_step_mm)

        # 4) 真机里这里替换成:
        #    move_flange_local(delta_q_a=du[0], delta_q_z=du[1])
        #    wait_settle()
        #    s_new = [measure_x_mid(), measure_endpoint_distance()]
        u_new = u + du
        s_new = plant(u_new)

        # 5) 用真实输入-输出变化更新 Jacobian
        ds = s_new - s
        J = broyden_update(J, du, ds)

        u = u_new
        s = s_new

    return pd.DataFrame(records)


def main():
    df = run_demo()

    print("\n=== Broyden dual-feature servo ===")
    print(df.to_string(index=False))

    final = df.iloc[-1]
    print("\nFinal:")
    print(f"  x_mid = {final['x_mid_mm']:.4f} mm")
    print(f"  L     = {final['L_mm']:.4f} mm")
    print(f"  q_a   = {final['q_a_mm']:.4f} mm")
    print(f"  q_z   = {final['q_z_mm']:.4f} mm")

    df.to_csv("broyden_demo_results.csv", index=False)

    # 每张图单独一个 figure，便于直接看趋势
    plt.figure()
    plt.plot(df["iteration"], df["x_mid_mm"], marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("x_mid (mm)")
    plt.title("x_mid convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("broyden_xmid.png", dpi=180)

    plt.figure()
    plt.plot(df["iteration"], df["L_mm"], marker="o")
    plt.axhline(80.0, linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Endpoint distance L (mm)")
    plt.title("Endpoint-distance convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("broyden_length.png", dpi=180)

    plt.figure()
    plt.plot(df["iteration"], df["q_a_mm"], marker="o", label="q_a")
    plt.plot(df["iteration"], df["q_z_mm"], marker="o", label="q_z")
    plt.xlabel("Iteration")
    plt.ylabel("Accumulated local translation (mm)")
    plt.title("Robot local translation commands")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("broyden_controls.png", dpi=180)

    plt.figure()
    plt.plot(df["iteration"], df["J_x_qa"], label="dx/dq_a")
    plt.plot(df["iteration"], df["J_x_qz"], label="dx/dq_z")
    plt.plot(df["iteration"], df["J_L_qa"], label="dL/dq_a")
    plt.plot(df["iteration"], df["J_L_qz"], label="dL/dq_z")
    plt.xlabel("Iteration")
    plt.ylabel("Estimated Jacobian entry")
    plt.title("Online Broyden Jacobian estimate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("broyden_jacobian.png", dpi=180)

    plt.show()


if __name__ == "__main__":
    main()
