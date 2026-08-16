"""
单点寄存器设置发送器。

移植自 main_T.py 的 FanucTpRegServoSender 类。
通过单点握手协议将位姿写入 FANUC PR 寄存器，
并检查机器人状态 DO 确认到位（模式1：单点焊接）。
"""

import threading
import time
import numpy as np

from .eip_io_thread import EIPIOThread
from .fifo_ring_sender import build_pr_from_template


class FanucTpRegServoSender:
    """
    配套 TP：REG_SERVO_L（单点握手一步到位）。

    发送器将单个位姿写入 FANUC PR 寄存器，设置命令寄存器
    触发 TP 程序执行运动，并等待到位确认。
    """

    def __init__(self, eip_io: EIPIOThread,
                 pr_cmd=10, r_cmd=110,
                 r_speed=120, r_ack=101,
                 r_state=102, r_fault=103,
                 do_ready=None,
                 min_speed=1, max_speed=200,
                 verbose=False):
        """
        Args:
            eip_io: EIPIOThread 实例
            pr_cmd: 命令 PR 寄存器编号
            r_cmd: 命令寄存器编号
            r_speed: 速度寄存器编号
            r_ack: 应答寄存器编号
            r_state: 状态寄存器编号
            r_fault: 故障寄存器编号
            do_ready: 数字输出编号（用于到位检查，可选）
            min_speed: 最小速度
            max_speed: 最大速度
            verbose: 是否打印调试信息
        """
        self.eip_io = eip_io
        self.pr_cmd = int(pr_cmd)
        self.r_cmd = int(r_cmd)
        self.r_speed = int(r_speed)
        self.r_ack = int(r_ack)
        self.r_state = int(r_state)
        self.r_fault = int(r_fault)
        self.do_ready = int(do_ready) if do_ready is not None else None
        self.min_speed = int(min_speed)
        self.max_speed = int(max_speed)
        self.verbose = bool(verbose)
        self._pr_template = None
        self._pr_template_lock = threading.Lock()

    def set_pr_template(self, template):
        """设置 PR 模板。"""
        with self._pr_template_lock:
            self._pr_template = list(template)

    # ── 底层 EIP 读写 ──────────────────────────
    def _get_r(self, idx, default=None):
        try:
            return int(self.eip_io.call("get_r", int(idx), timeout=0.2))
        except Exception:
            return default

    def _get_do(self, idx, default=None):
        try:
            return int(self.eip_io.call("get_do", int(idx), timeout=0.2))
        except Exception:
            return default

    def _set_r(self, idx, val):
        try:
            self.eip_io.call("set_r", int(idx), int(val), timeout=0.2)
            return True
        except Exception:
            return False

    def _set_pr(self, idx, pose6):
        try:
            with self._pr_template_lock:
                tpl = None if self._pr_template is None else list(self._pr_template)
                my_list = build_pr_from_template(pose6, tpl)
                self.eip_io.call("set_pr", int(idx), my_list, timeout=0.5)
                return True
        except Exception:
            return False

    # ── 状态读取 ───────────────────────────────
    def read_status(self):
        """读取控制器状态：应答、状态、故障。"""
        ack = self._get_r(self.r_ack, default=0)
        state = self._get_r(self.r_state, default=0)
        fault = self._get_r(self.r_fault, default=0)
        return ack, state, fault

    def check_do_ready(self):
        """检查数字输出到位信号。"""
        if self.do_ready is None:
            return True
        val = self._get_do(self.do_ready, default=0)
        return val != 0

    # ── 等待 ──────────────────────────────────
    def wait_ready(self, timeout=2.0, poll=0.01):
        """
        等待控制器空闲可接受新指令。

        state 语义:
          0 = 空闲
          1 = 运动中
          2 = 到位
         -1 = 故障
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            ack, state, fault = self.read_status()
            if state == -1:
                if self.verbose:
                    print(f"[REG] wait_ready: fault detected (fault={fault})")
                return False
            if ack == 0 and state != 1:
                return True
            time.sleep(poll)
        if self.verbose:
            print(f"[REG] wait_ready: timeout after {timeout}s (state={state})")
        return False

    def wait_done(self, timeout=500, poll=0.01):
        """
        等待运动到位。

        state 语义:
          2 = 到位（成功）
         -1 = 故障（失败）
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            ack, state, fault = self.read_status()

            # 检查 DO 到位信号（如果配置了）
            if self.do_ready is not None and self.check_do_ready():
                if self.verbose:
                    print("[REG] wait_done: DO ready signaled")
                return True

            if state == 2:
                if self.verbose:
                    print("[REG] wait_done: arrived (state=2)")
                return True
            if state == -1:
                if self.verbose:
                    print(f"[REG] wait_done: fault (fault={fault})")
                return False
            time.sleep(poll)
        if self.verbose:
            print(f"[REG] wait_done: timeout after {timeout}s")
        return False

    # ── 单点发送 ──────────────────────────────
    def send_single(self, pose6, utool=0, uframe=0,
                    speed_mm_s=10.0, clear_done=True,
                    done_clear_value=0, check_do=True):
        """
        发送单个位姿到控制器并等待执行完成。

        Args:
            pose6: [X, Y, Z, W, P, R] 位姿
            utool: 工具坐标系编号
            uframe: 用户坐标系编号
            speed_mm_s: 运动速度 (mm/s)
            clear_done: 完成后是否清除状态
            done_clear_value: 清除值
            check_do: 是否检查 DO 到位信号

        Returns:
            bool: 是否成功
        """
        pose6 = np.asarray(pose6, dtype=float).reshape(6)
        spd = int(np.clip(int(round(speed_mm_s)), self.min_speed, self.max_speed))

        if not self.wait_ready():
            if self.verbose:
                print("[REG] send_single: not ready")
            return False

        # 设置速度
        if not self._set_r(self.r_speed, spd):
            if self.verbose:
                print("[REG] send_single: set_r(speed) failed")
            return False

        # 写入目标位姿到 PR
        if not self._set_pr(self.pr_cmd, pose6):
            if self.verbose:
                print("[REG] send_single: set_pr failed")
            return False

        # 触发命令
        if not self._set_r(self.r_cmd, 1):
            if self.verbose:
                print("[REG] send_single: set_r(cmd) failed")
            return False

        if self.verbose:
            x, y, z, w, p, r = pose6
            print(f"[REG] send_single: pose=({x:.1f}, {y:.1f}, {z:.1f}, "
                  f"{w:.1f}, {p:.1f}, {r:.1f}) speed={spd}")

        # 等待到位
        if check_do:
            # 同时检查状态寄存器和 DO
            done = self.wait_done(timeout=500)
        else:
            done = self.wait_done(timeout=500)

        if not done:
            if self.verbose:
                print("[REG] send_single: timeout waiting for done")
            return False

        # 清除完成状态
        if clear_done:
            self._set_r(self.r_state, done_clear_value)

        return True

    def read_fault(self):
        """读取故障码。"""
        return self._get_r(self.r_fault, default=0)

    def clear_fault(self):
        """清除故障状态（命令字=0）。"""
        return self._set_r(self.r_cmd, 0)


class PcTrackAllStepSender:
    """Drive only the documented STEP branch of the existing PC_TRACK_ALL TP.

    Protocol used here:
      R[100] = 0 keeps FIFO mode disabled;
      PR[10] stores one Cartesian target;
      R[120] stores linear speed in mm/s;
      R[110] = 1 triggers the move;
      R[102] changes 0/2 -> 1 -> 2 (idle/done -> busy -> done).

    PC_TRACK_ALL has no R[101] acknowledgement and no R[103] fault word, so
    this class deliberately never reads or interprets either register.
    """

    def __init__(
        self,
        eip_io: EIPIOThread,
        *,
        pr_command: int = 10,
        r_fifo_command: int = 100,
        r_step_command: int = 110,
        r_step_state: int = 102,
        r_speed: int = 120,
        min_speed_mm_s: int = 1,
        max_speed_mm_s: int = 20,
    ):
        self.eip_io = eip_io
        self.pr_command = int(pr_command)
        self.r_fifo_command = int(r_fifo_command)
        self.r_step_command = int(r_step_command)
        self.r_step_state = int(r_step_state)
        self.r_speed = int(r_speed)
        self.min_speed_mm_s = int(min_speed_mm_s)
        self.max_speed_mm_s = int(max_speed_mm_s)

    def _get_r(self, register: int) -> int:
        return int(self.eip_io.call("get_r", int(register), timeout=1.0))

    def _set_r(self, register: int, value: int) -> None:
        self.eip_io.call(
            "set_r", int(register), int(value), timeout=1.0
        )

    def protocol_status(self) -> dict[str, int]:
        return {
            "fifo_command": self._get_r(self.r_fifo_command),
            "step_command": self._get_r(self.r_step_command),
            "step_state": self._get_r(self.r_step_state),
        }

    def assert_ready(self) -> dict[str, int]:
        status = self.protocol_status()
        if status["fifo_command"] != 0:
            raise RuntimeError(
                "PC_TRACK_ALL FIFO mode is not paused: expected R[100]=0, "
                f"got {status['fifo_command']}"
            )
        if status["step_command"] != 0:
            raise RuntimeError(
                "PC_TRACK_ALL STEP command is already pending: expected "
                f"R[110]=0, got {status['step_command']}"
            )
        if status["step_state"] not in (0, 2):
            raise RuntimeError(
                "PC_TRACK_ALL STEP branch is not idle/done: expected "
                f"R[102] in {{0,2}}, got {status['step_state']}"
            )
        return status

    @staticmethod
    def _position_matches(expected, actual, tolerance=1.0e-3) -> bool:
        # R-30iB normalizes the PR wire value 0 to 255 on readback.  These
        # fields do not select the active TP frame; PC_TRACK_ALL explicitly
        # sets UFRAME_NUM/UTOOL_NUM before executing PR[10].
        def equivalent_frame(first, second):
            pair = {int(first), int(second)}
            return len(pair) == 1 or pair == {0, 255}

        if not equivalent_frame(expected[0], actual[0]) or not equivalent_frame(
            expected[1], actual[1]
        ):
            return False
        return bool(
            np.allclose(
                np.asarray(expected[2:8], dtype=float),
                np.asarray(actual[2:8], dtype=float),
                atol=float(tolerance),
                rtol=0.0,
            )
        )

    def execute(
        self,
        pose6,
        *,
        template,
        utool: int = 0,
        uframe: int = 0,
        speed_mm_s: float = 5.0,
        timeout_s: float = 60.0,
        poll_s: float = 0.02,
    ) -> dict[str, int]:
        """Write one verified PR target, trigger STEP, and wait for R[102]=2."""
        self.assert_ready()
        target = build_pr_from_template(pose6, template)
        target[0] = int(utool)
        target[1] = int(uframe)
        speed = int(
            np.clip(
                int(round(speed_mm_s)),
                self.min_speed_mm_s,
                self.max_speed_mm_s,
            )
        )
        self._set_r(self.r_speed, speed)
        self.eip_io.call("set_pr", self.pr_command, target, timeout=2.0)
        readback = self.eip_io.call("get_pr", self.pr_command, timeout=2.0)
        if not self._position_matches(target, readback):
            raise RuntimeError("PR[10] readback does not match the requested target")

        # This is the only operation in this class that can start robot motion.
        self._set_r(self.r_step_command, 1)
        deadline = time.monotonic() + float(timeout_s)
        saw_busy = False
        while time.monotonic() < deadline:
            state = self._get_r(self.r_step_state)
            if state == 1:
                saw_busy = True
            elif state == 2:
                return {"step_state": state, "saw_busy": int(saw_busy), "speed": speed}
            elif state not in (0, 1, 2):
                raise RuntimeError(f"unexpected PC_TRACK_ALL R[102] state: {state}")
            time.sleep(float(poll_s))
        raise TimeoutError(
            f"PC_TRACK_ALL STEP timed out after {timeout_s:.1f}s"
        )
