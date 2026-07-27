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
