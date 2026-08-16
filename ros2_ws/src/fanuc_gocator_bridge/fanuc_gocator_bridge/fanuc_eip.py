from pycomm3 import CIPDriver
from pycomm3 import Services
import struct
import threading
from contextlib import AbstractContextManager


PR_VALUE_COUNT = 15
PR_BYTE_COUNT = 44


def decode_r_register(value) -> int:
    """Decode one FANUC numeric register as a signed 32-bit integer."""
    raw = bytes(value)
    if len(raw) < 4:
        raise RuntimeError(
            f"FANUC R register reply is too short: expected 4 bytes, got {len(raw)}"
        )
    return int.from_bytes(raw[:4], byteorder="little", signed=True)


def encode_r_register(value: int) -> bytes:
    """Encode one FANUC numeric register as a signed 32-bit integer."""
    return int(value).to_bytes(4, byteorder="little", signed=True)


def decode_cartesian_position(values):
    """Decode the 44-byte FANUC Cartesian/PR representation into scalars."""
    raw = bytes(values)
    if len(raw) < PR_BYTE_COUNT:
        raise RuntimeError(
            "FANUC Cartesian reply is too short: "
            f"expected {PR_BYTE_COUNT} bytes, got {len(raw)}"
        )
    return [
        int.from_bytes(raw[0:2], "little", signed=False),
        int.from_bytes(raw[2:4], "little", signed=False),
        *struct.unpack("<6f", raw[4:28]),
        *raw[28:32],
        *struct.unpack("<3f", raw[32:44]),
        list(raw),
    ]


def encode_cartesian_position(values) -> bytes:
    """Encode [UT, UF, X, Y, Z, W, P, R, turns..., ext...] for FANUC."""
    if len(values) < PR_VALUE_COUNT:
        raise ValueError(
            f"Cartesian position needs {PR_VALUE_COUNT} values, got {len(values)}"
        )
    return b"".join(
        [
            struct.pack("<H", int(values[0])),
            struct.pack("<H", int(values[1])),
            struct.pack("<6f", *(float(value) for value in values[2:8])),
            struct.pack("<4B", *(int(value) for value in values[8:12])),
            struct.pack("<3f", *(float(value) for value in values[12:15])),
        ]
    )


def return_joint_current_position(session):
    """Read FANUC controller display joints J1..J6 in degrees."""
    reply = session.generic_message(
        service=Services.get_attribute_single,
        class_code=0x7E,
        instance=0x01,
        attribute=0x01,
        data_type=None,
        connected=False,
        unconnected_send=False,
        route_path=True,
        name="fanucCURJPOSread",
    )
    values = list(reply.value)
    if len(values) < 28:
        raise RuntimeError(
            f"FANUC joint reply is too short: expected >=28 bytes, got {len(values)}"
        )
    return [
        struct.unpack("f", bytes(values[offset : offset + 4]))[0]
        for offset in range(4, 28, 4)
    ]


def decode_cartesian_current_position(values):
    """Decode FANUC CURPOS bytes without changing controller state."""
    values = list(values)
    if len(values) < 28:
        raise RuntimeError(
            "FANUC Cartesian reply is too short: "
            f"expected >=28 bytes, got {len(values)}"
        )
    return {
        "utool": int.from_bytes(bytes(values[0:2]), "little"),
        "uframe": int.from_bytes(bytes(values[2:4]), "little"),
        "x_mm": struct.unpack("<f", bytes(values[4:8]))[0],
        "y_mm": struct.unpack("<f", bytes(values[8:12]))[0],
        "z_mm": struct.unpack("<f", bytes(values[12:16]))[0],
        "w_deg": struct.unpack("<f", bytes(values[16:20]))[0],
        "p_deg": struct.unpack("<f", bytes(values[20:24]))[0],
        "r_deg": struct.unpack("<f", bytes(values[24:28]))[0],
    }


def return_cartesian_current_position(session):
    """Read active TCP pose, UTOOL and UFRAME from FANUC CURPOS."""
    reply = session.generic_message(
        service=Services.get_attribute_single,
        class_code=0x7D,
        instance=0x01,
        attribute=0x01,
        data_type=None,
        connected=False,
        unconnected_send=False,
        route_path=True,
        name="fanucCURPOSread",
    )
    return decode_cartesian_current_position(reply.value)


class FanucCIPSession(AbstractContextManager):
    """
        持久 EtherNet/IP (CIP) 会话：只建一次连接，反复发 explicit message。
        注意：pycomm3 的 driver 通常不是线程安全的，所以这里内置一把锁。
    """

    def __init__(self, drive_path: str, *, use_lock: bool = True):
        self.drive_path = drive_path
        self._driver = None
        self._lock = threading.RLock() if use_lock else None

    @property
    def driver(self):
        if self._driver is None:
            raise RuntimeError("CIP session not opened. Call open() first.")
        return self._driver

    def open(self):
        if self._driver is not None:
            return self

        d = CIPDriver(self.drive_path)

        # 兼容两种写法：有的版本提供 open()/close()，有的依赖 __enter__/__exit__
        if hasattr(d, "open"):
            d.open()
            self._driver = d
        else:
            # 用 context manager 的 enter 来打开底层会话
            self._driver = d.__enter__()

        return self

    def close(self, exc_type=None, exc=None, tb=None):
        if self._driver is None:
            return

        d = self._driver

        # 如果是 __enter__ 打开的，就走 __exit__
        if hasattr(d, "__exit__"):
            try:
                d.__exit__(exc_type, exc, tb)
            except TypeError:
                # 有的实现需要 3 个参数
                d.__exit__(exc_type, exc, tb)
        elif hasattr(d, "close"):
            d.close()

        self._driver = None

        # 让它可用 with FanucCIPSession(...) as s:

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close(exc_type, exc, tb)
        return False

        # 提供一个带锁的 generic_message 包装，后面所有读写都走这里

    def generic_message(self, **kwargs):
        if self._lock:
            with self._lock:
                return self.driver.generic_message(**kwargs)
        return self.driver.generic_message(**kwargs)


def returnCartesianCurrentPostion(drive_path,session: FanucCIPSession = None):
    if session is None:
        with CIPDriver(drive_path) as drive:
            myPRTag = drive.generic_message(
                service=Services.get_attribute_single,
                class_code=0x7D,
                instance=0x01,
                attribute=0x01,
                data_type=None,
                connected=False,
                unconnected_send=False,
                route_path=True,
                name='fanucCURPOSread'
            )
    else:
        myPRTag = session.generic_message(
            service=Services.get_attribute_single,
            class_code=0x7D,
            instance=0x01,
            attribute=0x01,
            data_type=None,
            connected=False,
            unconnected_send=False,
            route_path=True,
            name='fanucCURPOSread'
        )
    return decode_cartesian_position(myPRTag.value)


def readCartesianPositionRegister(drive_path, PRNumber, session: FanucCIPSession = None):
    if session is None:
        with CIPDriver(drive_path) as drive:
            myTag = drive.generic_message(
                service=Services.get_attribute_single,
                class_code=0x7B,
                instance=0x01,
                attribute=PRNumber,
                data_type=None,
                connected=False,
                unconnected_send=False,
                route_path=False,
                name='fanucPRSread'
            )
    else:
        myTag = session.generic_message(
            service=Services.get_attribute_single,
            class_code=0x7B,
            instance=0x01,
            attribute=PRNumber,
            data_type=None,
            connected=False,
            unconnected_send=False,
            route_path=False,
            name='fanucPRSread'
        )

    return decode_cartesian_position(myTag.value)


def writeCartesianPositionRegister(drive_path, PRNumber, myList, session: FanucCIPSession = None):
    myByteArray = encode_cartesian_position(myList)

    if session is None:
       with CIPDriver(drive_path) as drive:
           myTag = drive.generic_message(
               service=Services.set_attribute_single,
               class_code=0x7B,
               instance=0x01,
               attribute=PRNumber,
               data_type=None,
               connected=False,
               request_data=myByteArray,
               unconnected_send=False,
               route_path=False,
               name='fanucPRSwrite'
           )
    else:
        myTag = session.generic_message(
            service=Services.set_attribute_single,
            class_code=0x7B,
            instance=0x01,
            attribute=PRNumber,
            data_type=None,
            connected=False,
            request_data=myByteArray,
            unconnected_send=False,
            route_path=False,
            name='fanucPRSwrite'
       )

    return myTag.error


def writeR_Register(drive_path, RegNum, Value, session: FanucCIPSession = None):
    # 将Value变成 4 字节小端序。 当value=1时，会变成01 00 00 00，这 4 个字节就是这次“写 R 寄存器”的数据体。
    myBytes = encode_r_register(Value)

    if session is None:
        with CIPDriver(drive_path) as drive:
            myTag = drive.generic_message(
                service=0x10,              # 服务类型--写入
                class_code=0x6B,           # 要访问的是R寄存器这一类对象
                instance=0x1,              # 这一类对象里的固定实例
                attribute=RegNum,          # 写的是第 100 号 R 寄存器，也就是 R[100]
                request_data=myBytes[0:4], # 01 00 00 00  要写进去的值
                data_type=None,
                connected=False,
                unconnected_send=False,
                route_path=False,
                name='fanucDOread'
            )
    else:
        myTag = session.generic_message(
            service=0x10,
            class_code=0x6B,
            instance=0x1,
            attribute=RegNum,
            request_data=myBytes[0:4],
            data_type=None,
            connected=False,
            unconnected_send=False,
            route_path=False,
            name='fanucDOread'
        )

    """
    真正发出去的CIP内容结构
    [CIP Explicit Message]
        Service   : 0x10
        Class     : 0x6B
        Instance  : 0x01
        Attribute : 0x64   # 100 的十六进制
        Data      : 01 00 00 00
    """

    return myTag.error


def readR_Register(drive_path, RegNum,session: FanucCIPSession = None):
    if session is None:
        with CIPDriver(drive_path) as drive:
            myTag = drive.generic_message(
                service=0xe,
                class_code=0x6B,
                instance=0x1,
                attribute=RegNum,
                data_type=None,
                connected=False,
                unconnected_send=False,
                route_path=False,
                name='fanucRread'
            )
    else:
        myTag = session.generic_message(
            service=0xe,
            class_code=0x6B,
            instance=0x1,
            attribute=RegNum,
            data_type=None,
            connected=False,
            unconnected_send=False,
            route_path=False,
            name='fanucRread'
        )
    return decode_r_register(myTag.value)
