import glob

from setuptools import setup

setup(
    name="uvc_stereo",
    version="0.1.0",
    packages=["uvc_stereo"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/uvc_stereo"]),
        ("share/uvc_stereo", ["package.xml"]),
        ("share/uvc_stereo/config", glob.glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Junhyun Nam",
    maintainer_email="skawnsgus03@gmail.com",
    description="OV9281 USB 카메라 두 대(UNO Q 중계) → 정렬된 스테레오 토픽",
    license="MIT",
    entry_points={"console_scripts": ["uvc_stereo_bridge = uvc_stereo.bridge:main"]},
)
