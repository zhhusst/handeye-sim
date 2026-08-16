from setuptools import find_packages, setup


package_name = "fanuc_m20id25_support"


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
    ],
    install_requires=["setuptools", "numpy", "scipy"],
    zip_safe=True,
    maintainer="Zhang HaHa",
    maintainer_email="z@z.com",
    description="FANUC M-20iD/25 kinematics and frame conventions",
    license="Apache-2.0",
)
