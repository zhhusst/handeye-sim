import os
from glob import glob

from setuptools import find_packages, setup


package_name = "fanuc_gocator_bridge"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy", "scipy", "pycomm3"],
    zip_safe=True,
    maintainer="Zhang HaHa",
    maintainer_email="z@z.com",
    description="Safety-gated FANUC and Gocator backend for hand-eye calibration",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "fanuc_joint_state = fanuc_gocator_bridge.joint_state_node:main",
            "gocator_metric_adapter = fanuc_gocator_bridge.profile_metric_adapter_node:main",
            "measurement_sync = fanuc_gocator_bridge.measurement_sync_node:main",
            "fanuc_motion_bridge = fanuc_gocator_bridge.motion_bridge_node:main",
        ],
    },
)
