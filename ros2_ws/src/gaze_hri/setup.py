from glob import glob

from setuptools import find_packages, setup

package_name = "gaze_hri"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy", "pyyaml", "opencv-python"],
    zip_safe=True,
    maintainer="MoLBWA",
    maintainer_email="team@molbwa.local",
    description="Gaze-driven human-robot interaction for SO-ARM101",
    license="MIT",
    entry_points={
        "console_scripts": [
            "gaze_bridge = gaze_hri.gaze_bridge_node:main",
            "dwell_detector = gaze_hri.dwell_detector_node:main",
            "target_resolver = gaze_hri.target_resolver_node:main",
            "task_manager = gaze_hri.task_manager_node:main",
            "arm_server = gaze_hri.arm_server_node:main",
            "topdown_click = gaze_hri.topdown_click_node:main",
            "control_panel = gaze_hri.control_panel_node:main",
            "calibrate_world_to_base = gaze_hri.calibrate_world_to_base:main",
            "calib_tf_publisher = gaze_hri.calib_tf_publisher:main",
        ],
    },
)
