import importlib.util
import json
from pathlib import Path

import yaml


MODULE_PATH = Path("scripts/calibration_console.py")
SPEC = importlib.util.spec_from_file_location("calibration_console", MODULE_PATH)
CONSOLE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CONSOLE)


def test_trigger_response_and_status_fields_are_parseable():
    output = (
        "response:\n"
        "std_srvs.srv.Trigger_Response(success=True, "
        "message='state=MOVING; seeds=2/6; observation=SAFE')"
    )
    success, message = CONSOLE.extract_trigger_response(output)
    assert success
    assert CONSOLE.parse_key_values(message) == {
        "state": "MOVING",
        "seeds": "2/6",
        "observation": "SAFE",
    }


def test_status_parser_preserves_sequenced_motion_events():
    events = [{"seq": 1, "kind": "ROTATION_COMMAND", "target": "ry_negative"}]
    message = "state=MOVING; motion_events_json=" + json.dumps(events)
    output = (
        "response:\nstd_srvs.srv.Trigger_Response("
        f"success=True, message='{message}')"
    )

    success, parsed = CONSOLE.extract_trigger_response(output)

    assert success
    assert json.loads(CONSOLE.parse_key_values(parsed)["motion_events_json"]) == events


def test_progress_bar_is_bounded():
    assert CONSOLE.progress_bar(0, 6) == "[------------------]"
    assert CONSOLE.progress_bar(3, 6) == "[#########---------]"
    assert CONSOLE.progress_bar(9, 6) == "[##################]"


def test_seed_failure_event_distinguishes_rotation_from_servo_and_lists_causes():
    event = {
        "seq": 12,
        "kind": "FAILURE",
        "target": "ry_negative",
        "branch_index": 2,
        "branch_total": 8,
        "seed_count": 2,
        "angle_deg": 9.0,
        "stage": "SERVO",
        "from_deg": 8.0,
        "to_deg": 9.0,
        "failure_count": 2,
        "failure_limit": 3,
        "reason": "servo left the safe bilateral region",
        "measurement": "unsafe",
        "length_mm": 78.43,
        "margin_mm": -0.23,
        "retry_step_deg": 0.5,
        "recovery": "LOCAL_RETRY",
        "if_next_fails": "REFERENCE_RETRY",
        "history": [
            {"number": 1, "reason": "bilateral feature became unsafe"},
            {"number": 2, "reason": "servo left the safe bilateral region"},
        ],
        "detector_reason": "tracked_endpoint_topology_rejected",
    }

    rendered = CONSOLE.format_seed_motion_event(event, 6)

    assert "旋转分支 2/8" in rendered
    assert "失败 2/3（平移修正后）" in rendered
    assert "断点间距 78.43 mm、安全余量 -0.23 mm" in rendered
    assert "第1次：" in rendered and "第2次：" in rendered
    assert "拓扑/身份一致性检查" in rendered
    assert "若再失败，将回参考位姿以 0.50° 重试" in rendered


def test_terminal_failure_does_not_claim_another_microstep_retry():
    event = {
        "kind": "FAILURE",
        "target": "ry_negative",
        "branch_index": 2,
        "branch_total": 8,
        "seed_count": 2,
        "angle_deg": 8.0,
        "stage": "PROBE_BACK",
        "from_deg": 6.0,
        "to_deg": 8.0,
        "failure_count": 3,
        "failure_limit": 3,
        "reason": "dual-feature probing did not obtain three valid axes",
        "measurement": "missing",
        "recovery": "REFERENCE_RETRY",
        "if_next_fails": "",
    }

    rendered = CONSOLE.format_seed_motion_event(event, 6)

    assert "平移探测返回后" in rendered
    assert "先回上一个有效位姿，再回参考位姿" in rendered
    assert "若再失败" not in rendered


def test_noise_regime_exceedances_identify_the_modified_dominant_terms():
    noise = {
        "endpoint_gaussian_std_m": 0.0008,
        "robot_translation_std_m": 0.0005,
        "robot_rotation_std_deg": 0.03,
        "board_flatness_rms_m": 0.0005,
    }
    assert CONSOLE.noise_regime_exceedances(noise) == [
        ("断点提取", 10.0),
        ("机器人平移", 10.0),
        ("机器人旋转", 10.0),
        ("平板平面度", 50.0),
    ]


def test_disabled_direct_endpoint_noise_is_not_reported_as_active_stress():
    noise = {
        "direct_endpoint_injection_active": False,
        "endpoint_gaussian_std_m": 0.0008,
        "robot_translation_std_m": 0.000050,
        "robot_rotation_std_deg": 0.003,
        "board_flatness_rms_m": 0.000010,
    }

    assert CONSOLE.noise_regime_exceedances(noise) == []


def test_shared_shape_uses_its_validated_flatness_range():
    noise = {
        "endpoint_gaussian_std_m": 0.000080,
        "robot_translation_std_m": 0.000050,
        "robot_rotation_std_deg": 0.003,
        "board_flatness_rms_m": 0.0005,
    }
    assert CONSOLE.noise_regime_exceedances(
        noise, surface_model="shared"
    ) == []


def test_preflight_mode_prompt_supports_all_three_modes(monkeypatch):
    for entered, expected in (
        ("", "auto"),
        ("auto", "auto"),
        ("2", "always"),
        ("off", "off"),
    ):
        monkeypatch.setattr("builtins.input", lambda _prompt, value=entered: value)
        assert CONSOLE.ask_preflight_mode() == expected


def test_real_seed_excitation_targets_ten_degrees_and_rejects_weak_partials():
    payload = yaml.safe_load(
        Path(
            "ros2_ws/src/fanuc_gocator_bridge/config/real_calibration.yaml"
        ).read_text(encoding="utf-8")
    )
    seed = payload["/**"]["ros__parameters"]["seed"]
    assert seed["rotation_target_deg"] == 10.0
    assert seed["minimum_partial_rotation_deg"] == 5.0
    assert seed["minimum_rotation_separation_deg"] == 5.0


def test_complete_run_topic_set_covers_sensor_robot_detector_and_motion():
    topics = set(CONSOLE.FULL_RUN_RECORD_TOPICS)
    assert {
        "/gocator/profile_raw_mm",
        "/gocator/profile",
        "/fanuc/joint_states_raw",
        "/joint_states",
        "/calibration/flange_pose",
        "/calibration/endpoints",
        "/calibration/detection_control",
        "/calibration/seed_motion_state",
        "/profile_endpoint_detector/diagnostics",
        "/joint_trajectory_controller/follow_joint_trajectory/_action/status",
        "/rosout",
        "/tf",
        "/tf_static",
    } <= topics


def test_real_new_run_rearms_motion_between_seed_and_nbv(monkeypatch, tmp_path):
    console = CONSOLE.CalibrationConsole(backend="real")
    seed_file = tmp_path / "seeds.json"
    result_file = tmp_path / "calibration_result.json"
    dummy_node = object()
    arm_calls = []
    disarm_calls = []
    answers = iter((True, True))

    monkeypatch.setattr(console, "_start_seed_node", lambda *args: dummy_node)
    monkeypatch.setattr(CONSOLE, "wait_for_service", lambda *args: True)
    monkeypatch.setattr(
        CONSOLE,
        "call_trigger",
        lambda service, **kwargs: (True, "ok"),
    )
    monkeypatch.setattr(console, "_confirm_initial_pose", lambda *args: {})
    monkeypatch.setattr(
        CONSOLE,
        "ask_yes_no",
        lambda *args, **kwargs: next(answers),
    )
    monkeypatch.setattr(CONSOLE, "ask_integer", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        console,
        "_arm_real_motion",
        lambda: arm_calls.append("arm") or True,
    )
    monkeypatch.setattr(
        console,
        "_disarm_real_motion",
        lambda: disarm_calls.append("disarm"),
    )
    monkeypatch.setattr(console, "_stop_node", lambda *args: None)

    def finish_seeds(*_args):
        seed_file.write_text(
            json.dumps(
                {
                    "seeds": [{"label": f"seed_{index}"} for index in range(6)],
                    "rotation_diversity": {},
                }
            ),
            encoding="utf-8",
        )
        return True

    active_calls = []
    monkeypatch.setattr(console, "_monitor_automatic_seeds", finish_seeds)
    monkeypatch.setattr(console, "_print_seed_summary", lambda *args: None)
    monkeypatch.setattr(
        console,
        "_run_active",
        lambda *args, **kwargs: active_calls.append((args, kwargs)),
    )

    console._run_new_pipeline(
        "automatic", seed_file, result_file, tmp_path, "off"
    )

    assert arm_calls == ["arm", "arm"]
    assert disarm_calls == ["disarm", "disarm"]
    assert len(active_calls) == 1
