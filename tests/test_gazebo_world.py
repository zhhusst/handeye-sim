import xml.etree.ElementTree as ET
from pathlib import Path


WORLD_PATH = Path("ros2_ws/src/handeye_sim_bridge/config/handeye_world.sdf")


def test_simulation_world_disables_all_shadows():
    root = ET.fromstring(WORLD_PATH.read_text(encoding="utf-8"))
    world = root.find("world")
    assert world is not None
    assert world.attrib["name"] == "empty"
    assert world.findtext("scene/shadows") == "false"

    lights = world.findall("light")
    assert lights
    assert all(light.findtext("cast_shadows") == "false" for light in lights)


def test_ground_plane_is_invisible_but_keeps_collision():
    root = ET.fromstring(WORLD_PATH.read_text(encoding="utf-8"))
    ground = root.find("./world/model[@name='ground_plane']/link")
    assert ground is not None
    assert ground.find("collision/geometry/plane") is not None
    assert ground.find("visual") is None


def test_simulation_launcher_uses_project_world():
    launcher = Path("scripts/start_simulation.sh").read_text(encoding="utf-8")
    assert "WORLD_PATH=/workspace/ros2_ws/src/handeye_sim_bridge/config/handeye_world.sdf" in launcher
    assert 'gz sim -s -r -v "$GZ_VERBOSITY" "$WORLD_PATH" &' in launcher
    assert "'$WORLD_PATH'" in launcher
