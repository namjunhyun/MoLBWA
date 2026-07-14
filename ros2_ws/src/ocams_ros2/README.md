# ocams_ros2

oCamS-1MGN-U(스테레오 + 내장 IMU)용 ROS2 드라이버. Withrobot의 ROS1(catkin) 예제
([withrobot/oCamS](https://github.com/withrobot/oCamS) `Example/People_counter`)를 ROS2로 새로 포팅한
것 — 원본은 ROS2 Jazzy에서 빌드 자체가 안 됨(catkin/roscpp).

`orbslam3_ros2`의 stereo-inertial 노드가 구독하는 토픽 이름/형식에 맞춰 발행한다:

| 토픽 | 타입 | 내용 |
|------|------|------|
| `camera/left`, `camera/right` | `sensor_msgs/Image` (mono8) | 640×480, **정류(rectified)됨**, 30fps 실측 |
| `imu` | `sensor_msgs/Imu` | myAHRS+ AMGQUA ASCII 모드, 100Hz 실측 |

## 빌드 전 필요한 것 — Withrobot SDK 파일 (리포에 미포함)

`withrobot_camera.hpp/cpp`, `withrobot_utility.hpp/cpp`(GPLv3), `myahrs_plus.hpp`(Withrobot 자체
라이선스)는 이 노드가 의존하는 순수 C++/POSIX SDK인데, 라이선스 때문에 이 리포에는 커밋하지 않는다
(EyeTracker를 `external/`로 클론하는 것과 같은 이유).

```bash
git clone https://github.com/withrobot/oCamS.git /tmp/oCamS

SRC=/tmp/oCamS/Example/People_counter
DST=<이 디렉토리>   # ros2_ws/src/ocams_ros2

cp $SRC/include/withrobot_camera.hpp $SRC/include/withrobot_utility.hpp $SRC/include/myahrs_plus.hpp \
   $DST/include/ocams_ros2/
cp $SRC/src/withrobot_camera.cpp $SRC/src/withrobot_utility.cpp $DST/src/
```

**필수 수정 1곳**: `withrobot_camera.hpp`에 있는 `#include <ros/ros.h>` 줄을 삭제한다(ROS1 헤더라
ROS2 환경엔 없음 — 실제로는 디버그 매크로가 꺼져 있어 코드에서 안 쓰이므로 그냥 지우면 됨).

## 빌드 & 실행

```bash
colcon build --packages-select ocams_ros2
source install/setup.bash
ros2 run ocams_ros2 ocams_stereo_imu_node
```

## 하드웨어 메모
- USB ID `04b4:00f9`. `Withrobot::usb_device_info::product`가 **`"oCamS-1MGN-U"`**로 나온다 — 원본
  ROS1 코드는 `"oCamS-1CGN-U"`(컬러 모델명)만 매칭해서 이 유닛(모노 모델)에서는 장치를 못 찾았음.
  `ocams_stereo_imu_node.cpp`의 `enum_dev_list()`가 둘 다 매칭하도록 이미 고쳐져 있다.
- IMU는 `/dev/ttyACM0` (myAHRS+ virtual COM port, 115200 baud). udev 규칙 필요:
  ```
  ATTRS{idVendor}=="04b4", ATTRS{idProduct}=="00f9", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
  ```
  (`ID_MM_DEVICE_IGNORE`가 없으면 ModemManager가 가상 COM 포트를 잡아채서 연결이 불안정해질 수 있음.)
  `dialout` 그룹 추가도 안전빵으로 해두면 좋다(`sudo usermod -aG dialout $USER`, 재로그인 필요).

## IMU 단위 변환 — 삽질 기록

AMGQUA ASCII 문장은 **이미 물리 단위인 소수처럼 보이지만 사실 고정소수점 정수를 그대로 문자열로 보낸다**
(`myahrs_plus.hpp`의 `atof()` 파싱 자체엔 스케일링이 없어서, 정적으로만 보면 "이미 물리단위 decimal"이라고
착각하기 쉬움 — 실제로 이 프로젝트 세션에서 한 번 그렇게 잘못 "수정"했다가 원본 divisor가 맞다는 걸
raw 시리얼 캡처로 재검증하고 되돌렸다). 확인 방법:

```bash
stty -F /dev/ttyACM0 115200 raw -echo
timeout 2 head -c 300 /dev/ttyACM0 | xxd
```

- 쿼터니언 `/16384` → 크기 ≈ 1.0 (Q14 고정소수점, 검증됨)
- 가속도 `/100` → 정지 상태 크기 ≈ 9.8 m/s² (**이미 m/s² 단위**, 코드 주석의 "g 단위"라는 말은
  틀렸음 — divisor 자체는 맞음)
- 자이로 `/900` → **이미 rad/s 단위**로 직접 나옴

원본 ROS1 드라이버(`oCamS.cpp`)의 divisor(16384/100/900)를 그대로 썼다. 추가 단위 변환(×9.80665,
×π/180 등)을 넣으면 틀린다.
