"""
FIFO 环形队列发送器。

移植自 main_T.py 的 FanucTpFifoRingSender 类。
将位姿写入 FANUC 控制器的 PR 寄存器 FIFO 环形缓冲区，
供 TP 程序连续读取执行（模式2：连续焊接）。
"""

import threading
import time
import numpy as np

from .eip_io_thread import EIPIOThread


def build_pr_from_template(pose6, template_list):
    """
    基于模板构建 PR 写入数据。

    template_list: 来自 readCartesianPositionRegister() 的返回
    约定：template 的 [2..7] 是 X Y Z W P R
    """
    out = list(template_list)
    out[2] = float(pose6[0])
    out[3] = float(pose6[1])
    out[4] = float(pose6[2])
    out[5] = float(pose6[3])
    out[6] = float(pose6[4])
    out[7] = float(pose6[5])
    return out


def pose6_to_myList(pose6, utool=0, uframe=0,
                    turn=(0, 0, 0), bitflip=0,
                    ext=(0.0, 0.0, 0.0)):
    """
    pose6: [X, Y, Z, W, P, R] (mm,mm,mm,deg,deg,deg)

    myList 格式（EIP 写 PR 的函数按此 pack）:
    [UT, UF, X, Y, Z, W, P, R, Turn1, Turn2, Turn3, Bitflip, EXT0, EXT1, EXT2]
    """
    if pose6 is None or len(pose6) != 6:
        raise ValueError(f"pose6 must be len=6, got: {pose6}")

    X, Y, Z, W, P, R = pose6
    t1, t2, t3 = turn
    e0, e1, e2 = ext

    return [
        int(utool), int(uframe),
        float(X), float(Y), float(Z),
        float(W), float(P), float(R),
        int(t1), int(t2), int(t3), int(bitflip),
        float(e0), float(e1), float(e2),
    ]


class FanucTpFifoRingSender:
    """
    配套 TP：REG_SERVO_FIFO_*（控制器内 FIFO 环形队列连续 CNT）。

    发送器将位姿写入 PR 寄存器 FIFO 缓冲区，通过读写指针
    与 TP 程序同步，实现连续轨迹发送。
    """

    def __init__(self, eip_io: EIPIOThread,
                 pr_base=200, ring_size=200,
                 r_cmd=100, r_speed=120,
                 r_rd=202, r_wr=203,
                 low_watermark=None, high_watermark=None,
                 verbose=False, min_speed=1, max_speed=200):
        """
        Args:
            eip_io: EIPIOThread 实例
            pr_base: FIFO 基址 PR 编号
            ring_size: FIFO 环形队列大小
            r_cmd: 命令寄存器编号
            r_speed: 速度寄存器编号
            r_rd: 读指针寄存器编号
            r_wr: 写指针寄存器编号
            low_watermark: 低水位阈值（达到此值开始补点）
            high_watermark: 高水位阈值（补到此值停止）
            verbose: 是否打印调试信息
            min_speed: 最小速度
            max_speed: 最大速度
        """
        self.eip_io = eip_io
        self.pr_base = int(pr_base)
        self.N = int(ring_size)
        self.r_cmd = int(r_cmd)
        self.r_speed = int(r_speed)
        self.r_rd = int(r_rd)
        self.r_wr = int(r_wr)
        self.verbose = bool(verbose)
        self.min_speed = int(min_speed)
        self.max_speed = int(max_speed)
        self._last_enq_x = None
        self._last_rd = None
        self._pr_template = None
        self._pr_template_lock = threading.Lock()

        # 水位阈值（可从类外部修改）
        self.low_watermark = low_watermark if low_watermark is not None else (self.N // 2)
        self.high_watermark = high_watermark if high_watermark is not None else (self.N - 30)

    # ── 底层 EIP 读写 ──────────────────────────
    def _get_r(self, idx, default=None):
        try:
            return int(self.eip_io.call("get_r", int(idx), timeout=0.2))
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

    def set_pr_template(self, template):
        """设置 PR 模板（包含工具/用户坐标系等固定字段）。"""
        with self._pr_template_lock:
            self._pr_template = list(template)

    # ── FIFO 状态查询 ──────────────────────────
    def read_pointers(self):
        """读取当前读指针和写指针。"""
        rd = self._get_r(self.r_rd, default=0)
        wr = self._get_r(self.r_wr, default=0)
        rd = 0 if rd is None else int(rd) % self.N
        wr = 0 if wr is None else int(wr) % self.N
        return rd, wr

    def get_water_level(self):
        """返回当前 FIFO 填充量（水位）。"""
        rd, wr = self.read_pointers()
        return self.fill_level(rd, wr)

    def fill_level(self, rd=None, wr=None):
        """计算 FIFO 填充量。"""
        if rd is None or wr is None:
            rd, wr = self.read_pointers()
        return (wr - rd + self.N) % self.N

    # ── 控制命令 ───────────────────────────────
    def arm(self, speed_mm_s=100.0, reset_pointers=True, run=False):
        """
        初始化 FIFO 控制器。

        Args:
            speed_mm_s: 运动速度 (mm/s)
            reset_pointers: 是否清零读写指针
            run: 是否立即启动
        """
        spd = int(np.clip(int(round(speed_mm_s)), self.min_speed, self.max_speed))
        self._set_r(self.r_speed, spd)
        if reset_pointers:
            self._set_r(self.r_rd, 0)
            self._set_r(self.r_wr, 0)
        return self._set_r(self.r_cmd, 1 if run else 0)

    def run(self):
        """启动 FIFO 消费（命令字=1）。"""
        return self._set_r(self.r_cmd, 1)

    def stop(self):
        """停止 FIFO 消费（命令字=9）。"""
        return self._set_r(self.r_cmd, 9)

    # ── 发送 ──────────────────────────────────
    def send(self, pose6):
        """
        将一个位姿发送到 FIFO 环形队列。

        内部维护写指针，自动轮询 FIFO 水位。
        水位低于 low_watermark 时发送，补到 high_watermark 为止。

        Args:
            pose6: [X, Y, Z, W, P, R] 位姿

        Returns:
            bool: 是否成功写入
        """
        pose6 = np.asarray(pose6, dtype=float).reshape(6)

        # 读取当前 FIFO 状态
        rd, wr = self.read_pointers()
        fill = self.fill_level(rd, wr)

        # 如果水位 >= low_watermark，不补点
        if fill >= self.low_watermark:
            if self.verbose:
                pass  # 保持安静，避免日志洪流
            return False  # 未发送

        # 检查是否有空位
        next_wr = (wr + 1) % self.N
        if next_wr == rd:
            if self.verbose:
                print(f"[FIFO] send: FIFO full (rd={rd}, wr={wr})")
            return False  # full

        # 写入 PR 寄存器
        pr_idx = self.pr_base + wr
        ok_pr = self._set_pr(pr_idx, pose6)
        if not ok_pr:
            if self.verbose:
                print(f"[FIFO] send: set_pr failed at PR[{pr_idx}]")
            return False

        # 更新写指针
        ok_wr = self._set_r(self.r_wr, next_wr)
        if not ok_wr:
            if self.verbose:
                print("[FIFO] send: set_r(wr) failed")
            return False

        self._last_enq_x = float(pose6[0])
        self._last_rd = rd

        if self.verbose:
            print(f"[FIFO] send: wr {wr} -> {next_wr}, fill {fill} -> {fill+1}")
        return True

    def send_many(self, poses6):
        """
        批量发送多个位姿到 FIFO。

        Args:
            poses6: list/np.ndarray shape (K,6)

        Returns:
            int: 实际写入的点数
        """
        poses6 = np.asarray(poses6, dtype=float).reshape(-1, 6)

        rd, wr = self.read_pointers()
        free = (rd - wr - 1) % self.N  # ring buffer 空位数
        K = min(len(poses6), free)
        if K <= 0:
            return 0

        # 准备要写入的 PR 列表（PR号 + myList）
        items = []
        for i in range(K):
            pr_num = self.pr_base + ((wr + i) % self.N)
            with self._pr_template_lock:
                tpl = None if self._pr_template is None else list(self._pr_template)
                my_list = build_pr_from_template(poses6[i], tpl)
            items.append((pr_num, my_list))

        # 让 IO 线程执行批量写入
        self.eip_io.call("set_pr_batch", items, timeout=2.0)

        # 最后只写一次 WR
        next_wr = (wr + K) % self.N
        self.eip_io.call("set_r", int(self.r_wr), int(next_wr), timeout=0.2)

        return K

    def status(self):
        """返回当前 FIFO 状态字典。"""
        rd, wr = self.read_pointers()
        fill = self.fill_level(rd, wr)
        return {
            "rd": rd,
            "wr": wr,
            "fill": fill,
            "N": self.N,
            "watermark": fill,
        }

    def clear(self):
        """清零 FIFO（重置读写指针）。"""
        self._set_r(self.r_rd, 0)
        self._set_r(self.r_wr, 0)
        self._last_enq_x = None
        self._last_rd = None
        if self.verbose:
            print("[FIFO] clear: pointers reset to 0")
        return True
