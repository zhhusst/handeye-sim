import os
from glob import glob

from setuptools import find_packages, setup


package_name = "handeye_sim_backend"


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
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools", "numpy", "scipy"],
    zip_safe=True,
    maintainer="Zhang HaHa",
    maintainer_email="z@z.com",
    description="Gazebo raw-profile and truth-evaluation backend",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "scene_publisher = handeye_sim_backend.scene_publisher_node:main",
        ],
    },
)
