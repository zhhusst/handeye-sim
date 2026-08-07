import importlib.util
from pathlib import Path


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


def test_progress_bar_is_bounded():
    assert CONSOLE.progress_bar(0, 6) == "[------------------]"
    assert CONSOLE.progress_bar(3, 6) == "[#########---------]"
    assert CONSOLE.progress_bar(9, 6) == "[##################]"


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
