import numpy as np
import pytest
import yaml
from pathlib import Path

from fanuc_gocator_bridge.fanuc_eip import (
    decode_cartesian_position,
    decode_r_register,
    encode_cartesian_position,
    encode_r_register,
)
from fanuc_gocator_bridge.motion_safety import plan_small_linear_move
from fanuc_gocator_bridge.reg_sender import PcTrackAllStepSender


def test_launch_arguments_are_not_shadowed_by_node_specific_yaml():
    config = yaml.safe_load(
        Path(
            "/workspace/ros2_ws/src/fanuc_gocator_bridge/config/real_calibration.yaml"
        ).read_text(encoding="utf-8")
    )
    motion = config["fanuc_motion_bridge"]["ros__parameters"]
    assert {"mode", "motion_writes_enabled", "robot_ip"}.isdisjoint(motion)
    joints = config["fanuc_joint_state"]["ros__parameters"]
    assert "robot_ip" not in joints
    gocator = config["gocator_profile_driver"]["ros__parameters"]
    assert "sensor_ip" not in gocator


def test_fanuc_register_is_signed_32_bit_not_first_byte():
    for value in (0, 1, 255, 3600, 23000, -1):
        assert decode_r_register(encode_r_register(value)) == value


def test_fanuc_pr_roundtrip_preserves_uf_ut_and_scalars():
    values = [
        1,
        1,
        961.234,
        -229.862,
        248.643,
        179.513,
        -22.389,
        -0.552,
        1,
        0,
        2,
        7,
        0.0,
        0.0,
        0.0,
    ]
    decoded = decode_cartesian_position(encode_cartesian_position(values))
    assert decoded[:2] == [1, 1]
    assert np.allclose(decoded[2:8], values[2:8], atol=1.0e-4)
    assert decoded[8:12] == values[8:12]


def test_base_to_controller_pose_matches_validated_stationary_sample():
    # Raw J3=-46.7173 deg; validated URDF J3=raw J3+J2=-38.9809 deg.
    current = np.deg2rad(
        [-13.863134, 7.736391, -38.980917, -12.353806, -21.791998, 23.825813]
    )
    target = current.copy()
    target[5] += np.deg2rad(1.0)
    plan = plan_small_linear_move(current, target)
    # Translation by [0,0,425] mm is exactly the controller/base convention
    # validated against UF1/UT1 CURPOS on the real robot.
    assert np.allclose(
        plan_small_linear_move(current, current).target_pose_xyz_wpr[:3],
        [961.2342, -229.8624, 248.6433],
        atol=0.001,
    )
    assert plan.maximum_joint_step_deg == pytest.approx(1.0)


class _FakeEipIo:
    def __init__(self):
        self.calls = []
        self.registers = {100: 0, 110: 0, 102: 0, 120: 0}
        self.pr = {}
        self._state_reads_after_trigger = 0

    def call(self, operation, *args, timeout=None):
        self.calls.append((operation, args))
        if operation == "get_r":
            register = args[0]
            if register == 102 and self.registers[110] == 1:
                self._state_reads_after_trigger += 1
                return 1 if self._state_reads_after_trigger == 1 else 2
            return self.registers[register]
        if operation == "set_r":
            register, value = args
            self.registers[register] = value
            return True
        if operation == "set_pr":
            register, values = args
            self.pr[register] = list(values)
            return True
        if operation == "get_pr":
            values = list(self.pr[args[0]])
            # Match the observed R-30iB normalization on PR readback.
            if values[0] == 0:
                values[0] = 255
            if values[1] == 0:
                values[1] = 255
            return values
        raise AssertionError(operation)


def test_pc_track_step_protocol_writes_pr_before_trigger_and_uses_no_fake_ack():
    fake = _FakeEipIo()
    sender = PcTrackAllStepSender(fake)
    template = [1, 1, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0]
    sender.execute(
        [100, 200, 300, 170, -20, 5],
        template=template,
        speed_mm_s=5,
        timeout_s=0.2,
        poll_s=0.0,
    )
    trigger_index = fake.calls.index(("set_r", (110, 1)))
    pr_index = next(
        index
        for index, call in enumerate(fake.calls)
        if call[0] == "set_pr"
    )
    assert pr_index < trigger_index
    assert fake.pr[10][:2] == [0, 0]
    touched_registers = {
        args[0]
        for operation, args in fake.calls
        if operation in {"get_r", "set_r"}
    }
    assert touched_registers <= {100, 102, 110, 120}
