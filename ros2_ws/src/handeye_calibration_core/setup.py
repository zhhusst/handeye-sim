from setuptools import find_packages, setup


package_name = "handeye_calibration_core"


setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "scipy", "PyYAML"],
    zip_safe=True,
    maintainer="Zhang HaHa",
    maintainer_email="z@z.com",
    description=(
        "ROS-independent bilateral-corner active hand-eye calibration core"
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "calibration_demo = calibration_pipeline.cli:main",
        ],
    },
)
