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
        ("机器人平移", 0.0005 / 0.000030),
        ("机器人旋转", 10.0),
        ("平板平面度", 0.0005 / 0.000030),
    ]
