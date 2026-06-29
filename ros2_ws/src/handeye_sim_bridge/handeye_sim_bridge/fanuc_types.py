"""FANUC M-20iD/25 运动学数据类型

从 backup/robot_weld_zhh/utils/TypeDefinition.py 精简
"""
from dataclasses import dataclass
from typing import List


@dataclass
class RobotParam:
    """FANUC M-20iD/25 DH 参数容器"""
    a: List[float]  # 连杆长度 (m)
    d: List[float]  # 连杆偏距 (m)
