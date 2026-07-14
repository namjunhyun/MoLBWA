# 09. 시각화 — 무엇으로, 어떻게 보여줄 것인가

융합(docs/03)의 결과물 `p_W`(시선이 향한 세계 3D점)를 **실시간으로 보여주는** 레이어.
"돌아간다"와 "보여줄 수 있다"는 다른 문제이고, 이 문서는 후자를 다룬다.

---

## 1. 왜 ORB-SLAM3 기본 뷰어로는 부족한가

ORB-SLAM3의 Pangolin 뷰어(및 RViz의 PointCloud2)가 그리는 건 **sparse map point** —
트래킹용 랜드마크지 사람이 보라고 만든 게 아니다. 점 몇 천 개로는:

- 시선이 **어느 표면**에 닿았는지 알 수 없다 (레이캐스팅할 표면이 없음).
- 눈 영상 / 씬 영상 / 3D를 **한 타임라인에서** 볼 수 없다.
- 시선이 튈 때 **동공 검출이 깨진 건지 SLAM이 깨진 건지** 구분할 방법이 없다.

즉 뷰어를 바꾸는 게 아니라 **맵 표현과 화면 구성을 바꿔야** 한다.

## 2. 툴 선정: Rerun

| 레이어 | 툴 | 비고 |
|---|---|---|
| 시각화 | **Rerun** (`pip install rerun-sdk`) | 3D + 영상 + 시계열을 한 타임라인에 |
| 표면(mesh) | Open3D `VoxelBlockGrid` (GPU TSDF) | 히트맵을 얹을 캔버스 + 레이캐스팅 fallback |
| 스테레오 깊이 | OpenCV `cuda::StereoSGM` | 실시간 30Hz. 부족하면 나중에 신경망으로 교체 |
| 글루 | ROS 2 | 서로 다른 주기·프로세스(C++/Python) 결합, `tf2` |

**Rerun을 고른 이유**는 RViz보다 예뻐서가 아니라, 이 프로젝트가 **이종 데이터의 시간 정렬**
문제이기 때문이다. Rerun은 눈 IR 영상 · 씬 영상 · gaze ray · SLAM 궤적 · 스칼라 지표를
하나의 타임라인에 얹고 **되감을 수 있다**. "그 순간 시선이 왜 튀었나"를 사후에 볼 수 있는지가
디버깅 속도를 가른다.

Rerun은 ROS 2를 대체하지 않는다. **RViz 자리에만** 들어간다.

---

## 3. 하드웨어 변경이 시각화에 미치는 영향 ⚠️

눈 카메라가 **OV9281 → GC0308**로 바뀌었다. 검출은 확인됐지만, 시각화 설계의 **전제가 바뀐다.**

| | OV9281 (당초 계획) | **GC0308 (실제)** |
|---|---|---|
| 셔터 | 글로벌 | **롤링** |
| 해상도/속도 | 1280×800, 120fps | **640×480, 30fps** |
| 센서 | 모노 | **컬러(Bayer)** |

### 이게 무슨 뜻인가

**(1) 사케이드(saccade)는 못 잡는다. → fixation 기반 시각화로 확정.**
사케이드는 수십 ms에 최대 500°/s로 지나간다. 30fps(33ms/프레임)로는 원리상 몇 샘플 못 얻는다.
따라서 **scanpath의 미세 구조나 사케이드 속도 분석은 이 하드웨어의 스펙 밖**이다.
대신 **fixation(응시)** — 어디를 얼마나 오래 봤는가 — 는 충분히 잡힌다.
시각화 목표를 **"3D fixation 히트맵"**으로 명확히 잡고, 그 이상을 약속하지 않는다.

**(2) 30Hz는 오히려 동기화를 단순하게 만든다.**
당초 걱정(gaze 120Hz vs pose 30Hz의 보간)이 사라진다. gaze와 pose가 같은 30Hz대라
ROS 2 `message_filters`의 `ApproximateTimeSynchronizer`로 충분하다.

**(3) 롤링셔터 왜곡은 고정 중엔 무시 가능, 급격한 머리 회전 중엔 아니다.**
동공이 화면에서 빠르게 움직일 때 타원이 기울어 검출 오차가 된다.
→ **머리 회전 각속도(IMU)가 큰 구간의 gaze 샘플은 신뢰도를 낮추거나 버린다.**
Rerun 시계열에 각속도를 같이 찍어두면 이 상관관계가 눈에 보인다.

**(4) 컬러 센서라 850nm IR 감도가 낮다.**
노출을 올리면 모션 블러가 늘어 (1)(3)을 악화시킨다. 노출·게인은 시각화 품질과 직결되므로
**IR 조명을 밝게 해서 노출 시간을 짧게** 유지하는 방향이 맞다.

---

## 4. 화면 구성

| 뷰 | 엔티티 | 목적 |
|---|---|---|
| **3D** | 카메라 궤적, TSDF mesh, gaze ray, fixation 히트맵 | 본 화면 |
| **씬 영상** | oCamS 좌영상 + gaze 커서 `(u,v)` | 현재 `gaze_on_scene.py`가 하는 것 |
| **눈 영상** | GC0308 IR + 동공 타원 | **필수** — 검출 실패와 SLAM 실패를 구분하는 유일한 수단 |
| **시계열** | gaze 벡터 std, 깊이 D, IMU 각속도, ORB 트래킹 상태, `model_centers` | 이상 원인 추적 |

### 로깅 스케치

```python
import rerun as rr
rr.init("molbwa", spawn=True)

rr.set_time_nanos("t", ts_ns)                      # 모든 스트림의 공통 시간축

# 머리 pose (ORB-SLAM3) — 하위 엔티티가 자동으로 따라온다
rr.log("world/cam", rr.Transform3D(translation=T_WS[:3, 3], mat3x3=T_WS[:3, :3]))
rr.log("world/cam/img", rr.Pinhole(image_from_camera=K, resolution=[SW, SH]))
rr.log("world/cam/img/rgb",  rr.Image(scene_left))
rr.log("world/cam/img/gaze", rr.Points2D([[u, v]], radii=12))   # 지금의 초록 원

# 융합 결과 (docs/03)
rr.log("world/gaze/ray", rr.Arrows3D(origins=[T_WS[:3, 3]], vectors=[p_W - T_WS[:3, 3]]))
rr.log("world/gaze/hit", rr.Points3D([p_W], radii=[r_uncert]))  # ↓ 5절

# 디버그
rr.log("eye/ir", rr.Image(eye_frame))
rr.log("plot/depth",   rr.Scalar(D))
rr.log("plot/gyro",    rr.Scalar(omega_norm))   # 롤링셔터 신뢰도 게이트
rr.log("plot/gaze_std", rr.Scalar(gaze_std))    # notes/2026-07-02의 std≈0.18 추적
```

---

## 5. 시선을 점으로 찍지 말 것

**각도 오차 1.5°는 3m 거리에서 약 8cm다.** `p_W`를 점 하나로 그리면 실제보다 훨씬 정확해
보이는 거짓말이 된다. 깊이에 비례해 커지는 **불확실성 반경**으로 그린다.

```python
r_uncert = D * np.tan(np.deg2rad(theta_err))   # theta_err = 캘리브레이션 잔차에서 실측
```

fixation 히트맵도 점 누적이 아니라 **mesh vertex에 거리 기반 가우시안 커널을 누적**하면
오차 특성이 자연스럽게 반영된다. 정직하면서 결과물도 더 설득력 있다.

---

## 6. 시각화 전에 반드시 선행되어야 할 것 (현재 코드 기준)

`src/gaze_on_scene.py`는 2D 오버레이까지는 동작하지만, **3D 실시간 시각화를 얹으면 아래가 터진다.**

### (1) `gaze_vector.txt` 파일 IPC 제거 — 1순위
지금 `tracker.process_frame()`이 텍스트 파일에 쓰고 `read_last_gaze()`가 다시 읽는다.
데모에선 돌아가지만, pose·깊이와 **타임스탬프를 맞춰야 하는 순간** 레이스 컨디션과
프레임 드랍이 된다. `Orlosky3DEyeTracker.process_frame`이 **벡터를 리턴하도록** 고쳐
메모리로 받는다.

### (2) 씬 카메라 캘리브레이션 실측
현재 `fx = SW * 0.94` **근사값**. 2D 원만 그릴 땐 티가 안 나지만,
**깊이를 곱해 3D 점을 만드는 순간 미터 단위 오차로 증폭된다.** oCamS 스테레오 캘리브(K, D, baseline) 필수.

### (3) 우영상 추출
`scene_left()`가 YUYV의 Y채널(좌)만 쓴다. 스테레오 깊이를 얻으려면 UV채널의 우영상도 꺼내야 한다.
(oCamS-1MGN-U는 하드웨어 동기화 글로벌셔터 — 이 부분은 유리하다.)

### (4) 1점 캘리브레이션의 구조적 한계
`rotation_from_a_to_b`는 **회전 R만** 구하고 눈↔씬 카메라의 **translation(수 cm)을 무시**한다.
→ 캘리브레이션한 거리에서만 맞고 **다른 거리에서는 어긋난다** (시차 오차, parallax).

2D 오버레이에선 안 보이다가, 3D 융합에서 *"물체를 응시하는데 `p_W`가 엉뚱한 곳에 찍힌다"*로 터진다.
docs/03의 검증 항목(*pose를 흔들며 같은 물체 응시 → `p_W`가 월드에서 고정되는가*)이 정확히 이걸 잡아낸다.

**대응**: 여러 거리(0.5m / 1.5m / 3m)에서 캘리브 점을 모아 **R과 t를 함께** 푼다.

---

## 7. 진행 순서

- [ ] **A. 파일 IPC 제거 + Rerun 배선** — 눈/씬 2D를 먼저 Rerun으로 옮긴다 (기능 동일, 배선만 교체)
- [ ] **B. oCamS 스테레오 캘리브 + 우영상 + SGM 깊이** — 가장 오래 걸림. 여기서 `D`가 나온다
- [ ] **C. ORB-SLAM3(stereo-inertial) pose 연결** → `T_WS`
- [ ] **D. 융합** `p_W` + 불확실성 반경 → Rerun 3D
- [ ] **E. 다중거리 캘리브레이션 (R+t)** — docs/03 검증(`p_W` 월드 고정성) 통과가 합격선
- [ ] **F. TSDF mesh + fixation 히트맵 누적**

A는 반나절, B가 병목. **C·D는 B 없이는 의미가 없다** (깊이 없으면 3D 점이 안 나온다).

## 8. 이 하드웨어로 약속할 수 있는 것 / 없는 것

**할 수 있다**: 실시간 3D fixation 위치, 시선 광선, 응시 히트맵, 머리 궤적과의 결합.
**할 수 없다**: 사케이드 궤적, 마이크로새케이드, 100Hz급 시선 동역학. — GC0308 30fps의 물리적 한계.

과제 발표·데모의 서술을 **전자에 맞춘다.**
