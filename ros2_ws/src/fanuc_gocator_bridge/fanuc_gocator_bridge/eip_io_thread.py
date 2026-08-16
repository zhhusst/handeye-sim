"""
EIP IO Thread — 独占EtherNet/IP会话的单线程执行器。

移植自 main_T.py 的 EIPIOThread 类。
通过内部 queue.Queue 接收操作请求，由工作线程顺序执行。
"""

import threading
import queue
import time

from . import fanuc_eip as EIP


class EIPIOThread(threading.Thread):
    """
    独占的 EtherNet/IP IO 线程。

    - 维护一个持久 FanucCIPSession
    - 其它线程通过 call() 提交读写请求
    - 支持操作: set_pr / get_pr / set_r / get_r / get_curpos / set_pr_batch / get_do
    """

    def __init__(self, session: EIP.FanucCIPSession):
        super().__init__(daemon=True)
        self.session = session
        self.q = queue.Queue(maxsize=5000)
        # Do not shadow threading.Thread._stop(), otherwise join() fails.
        self._stop_event = threading.Event()

    def stop(self):
        """安全停止 IO 线程。"""
        self._stop_event.set()
        try:
            self.q.put_nowait(("__stop__", (), {}, None))
        except queue.Full:
            pass

    def call(self, op: str, *args, timeout=0.5, **kwargs):
        """
        向 IO 线程提交一个操作并等待结果。

        Args:
            op: 操作名 ('set_pr', 'get_pr', 'set_r', 'get_r', 'get_curpos',
                      'set_pr_batch', 'get_do')
            *args: 操作参数
            timeout: 等待响应的超时时间（秒）
            **kwargs: 额外关键字参数

        Returns:
            操作结果

        Raises:
            RuntimeError: 操作失败时抛出
            queue.Empty: 超时时抛出
        """
        resp_q = queue.Queue(maxsize=1)
        self.q.put((op, args, kwargs, resp_q))
        ok, payload = resp_q.get(timeout=timeout)
        if not ok:
            raise RuntimeError(payload)
        return payload

    def run(self):
        """线程主循环：消费队列中的操作请求。"""
        while not self._stop_event.is_set():
            try:
                op, args, kwargs, resp_q = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            if op == "__stop__":
                break

            t0 = time.perf_counter()
            try:
                if op == "set_pr":
                    pr_num, my_list = args
                    err = EIP.writeCartesianPositionRegister(
                        self.session.drive_path, pr_num, my_list,
                        session=self.session
                    )
                    if err:
                        raise RuntimeError(f"set_pr error={err}")
                    result = True

                elif op == "get_pr":
                    (pr_num,) = args
                    result = EIP.readCartesianPositionRegister(
                        self.session.drive_path, pr_num,
                        session=self.session
                    )

                elif op == "set_r":
                    r_num, val = args
                    err = EIP.writeR_Register(
                        self.session.drive_path, r_num, val,
                        session=self.session
                    )
                    if err:
                        raise RuntimeError(f"set_r error={err}")
                    result = True

                elif op == "get_r":
                    (r_num,) = args
                    result = EIP.readR_Register(
                        self.session.drive_path, r_num,
                        session=self.session
                    )

                elif op == "get_curpos":
                    result = EIP.returnCartesianCurrentPostion(
                        self.session.drive_path,
                        session=self.session
                    )

                elif op == "set_pr_batch":
                    (items,) = args  # items = [(pr_num, myList), ...]
                    for pr_num, my_list in items:
                        err = EIP.writeCartesianPositionRegister(
                            self.session.drive_path, int(pr_num),
                            my_list, session=self.session
                        )
                        if err:
                            raise RuntimeError(
                                f"set_pr_batch error at PR={pr_num}, err={err}"
                            )
                    result = True

                elif op == "get_do":
                    # 读取数字输出 (Digital Output) 状态
                    (do_num,) = args
                    result = EIP.readR_Register(
                        self.session.drive_path, do_num,
                        session=self.session
                    )

                else:
                    raise ValueError(f"Unknown op: {op}")

                if resp_q is not None:
                    resp_q.put((True, result))

            except Exception as e:
                if resp_q is not None:
                    resp_q.put((False, str(e)))

            # 可选性能日志（取消注释以启用）
            # dt_ms = (time.perf_counter() - t0) * 1000
            # if dt_ms > 20:
            #     print(f"[EIPIO] {op} took {dt_ms:.1f}ms")
