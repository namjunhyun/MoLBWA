# 12. 눈 카메라 ↔ 씬 카메라 R+t 캘리브레이션 (설계, 미구현)

## 배경

`gaze_on_scene.py`의 현재 다점 캘리브(`m` 키)는 **회전(R) + 초점거리(fx) 스케일**만 푸는
구조다 (Wahba's problem / SVD). 시선 벡터쌍을 "원점이 같다"고 가정하고 방향만 맞추기 때문에,
눈 카메라와 씬 카메라 사이의 실제 물리적 평행이동(t, 수 cm)을 설명하지 못한다. 캘리브한
거리에서는 잘 맞지만 다른 거리를 보면 시차(parallax) 오차로 어긋난다 —
`docs/09_visualization.md` §6-(4)에서 지적된 한계.

목표: 여러 거리(0.5m / 1.5m / 3m 등)에서 캘리브 점을 모아 **R과 t를 동시에** 푼다.

## 막다른 길: 트래커의 gaze_origin은 못 쓴다

처음엔 "Orlosky 트래커가 `gaze_origin`(sphere_center)도 주니까, 눈알 회전 중심을 고정점으로
놓고 (R, t) 6자유도를 풀면 되지 않나" 생각했는데, `external/EyeTracker/3DTracker/Orlosky3DEyeTracker.py`의
`compute_gaze_vector()`를 직접 읽어보니 아니었다.

`sphere_center`는 물리 단위 3D 위치가 아니라, **눈 카메라 화면 안에서 눈이 보이는 2D 위치를
그래픽스용 합성 NDC 좌표(z=0, 임의 스케일 1.5)로 옮긴 값**이다 (`camera_position = [0,0,3]`,
`far_clip=100` 같은 하드코딩된 렌더링 파라미터 기반). 즉 사람 머리가 눈 카메라에 대해
살짝만 움직여도 이 값이 흔들리고, 실제 세계 단위(m)와 대응이 안 된다. 이걸 씬 카메라의
실측 3D 좌표(m)와 같은 최적화에 섞으면 물리적으로 의미 없는 자유도가 섞여 들어간다.

**결론**: 눈 원점은 트래커에서 받아오지 말고, **캘리브레이션 데이터 자체에서 미지수로 같이
푼다.** 지금 파이프라인이 이미 안정적으로 주는 건 시선 **방향**(`smooth_dir`, 단위벡터)뿐이고,
이거면 충분하다.

## 수식

미지수: 회전 `R`(3자유도, Rodrigues 벡터로 파라미터화), 눈의 유효 원점 `p_eye`(씬 카메라
좌표계 기준 3D 위치, 3자유도) — 총 6개.

캘리브 점 `i`마다 갖고 있는 것:
- `d_i`: 그 순간의 시선 방향 (눈 카메라 좌표계, 단위벡터) — 지금 `smooth_dir`로 이미 있음.
- `X_i`: 실제로 응시한 물체의 3D 위치 (씬 카메라 좌표계, 미터 단위) —
  `X_i = D_i · K⁻¹ [u_i, v_i, 1]ᵀ` (`fusion.py`의 공식과 동일). `(u_i, v_i)`는 클릭한 픽셀,
  `D_i`는 그 지점의 depth.

풀 것: 광선 `(원점=p_eye, 방향=R·d_i)`가 `X_i`를 최대한 가깝게 지나가도록 `R`, `p_eye`를
비선형 최소제곱으로 동시에 푼다 (`scipy.optimize.least_squares`). 잔차는 점-직선 거리
(미터 단위라 오차를 물리적으로 바로 해석 가능):

```
residual_i = || (X_i - p_eye) - ((X_i - p_eye)·(R·d_i)) (R·d_i) ||
```

초기값: `R0`는 기존 방식(고정 `p_eye0=0` 가정, Wahba)으로 구한 회전, `p_eye0 = [0,0,0]`
(CAD에 눈 카메라 위치가 없어서 실측 초기 추정치가 없음, 자로 대략 재서 넣어도 됨).

점 개수: 최소 6~9개 이상 권장(3거리 × 여러 위치). 너무 적으면 R/t가 서로 얽혀 잘 안 풀린다.

## Depth 획득 — 자로 재는 대신 스테레오 자동 측정

`docs/09_visualization.md` §7의 미완료 항목 "B. oCamS 스테레오 캘리브 + 우영상 + SGM 깊이"를
여기서 같이 해결한다. 이미 있는 재료:

- `src/ocams_calib.py`가 `left_opencv.yaml`/`right_opencv.yaml`(2026-08-11 재캘리브레이션된
  값, `docs/11_camera_imu_calibration.md` 참고)로 `RECTIFIED_K`, `BASELINE_M`,
  `build_rectify_maps()`(left/right 둘 다 반환)를 이미 계산해서 갖고 있다.
- 다만 `gaze_on_scene.py`의 `scene_left()`는 **왼쪽만** rectify해서 쓰고, `right_maps`는
  `build_rectify_maps()`가 반환해도 지금 코드에서 버려진다 (`_right_maps_unused` 변수명 참고).

필요한 추가 작업:
1. `scene_left()`와 대응하는 `scene_right()` 추가 (YUYV의 UV 채널 → 우영상, `right_maps`로
   rectify).
2. `cv2.StereoSGBM`으로 disparity 계산 (오늘 rectification 검증 때 썼던 파라미터가 출발점:
   `numDisparities=128, blockSize=7, P1/P2=8·9·bs², 32·9·bs², disp12MaxDiff=1,
   uniquenessRatio=10, speckleWindowSize=100, speckleRange=2` — `2026-08-11` 세션에서
   32% 유효 픽셀 확인).
3. 클릭한 픽셀 주변 작은 윈도우(예 5x5)의 disparity 중앙값을 사용(노이즈 완화), 유효 disparity가
   없으면 그 점은 버리고 다시 클릭하라고 안내.
4. `D = fx · baseline / disparity` (`ocams_calib.RECTIFIED_K[0,0]`, `ocams_calib.BASELINE_M`).

## 데이터 수집 절차 (구현되면)

1. `gaze_on_scene.py`에 새 모드(가칭 `M`, 기존 `m`과 구분)로 추가 — 클릭 시 방향뿐 아니라
   그 픽셀의 SGBM depth도 같이 기록.
2. 물체를 0.5m / 1.5m / 3m 각 거리에 두고, 화면 중앙 + 네 모서리 근처 등 여러 위치를 골고루
   응시 + 클릭 (총 9개 이상 목표).
3. 점이 6개 이상 모이면 `scipy.optimize.least_squares`로 `(R, p_eye)` 재계산, 잔차(미터)를
   화면에 표시.
4. `docs/03_fusion.md` 체크리스트의 "pose 이동 시 `p_W` 월드 고정성" 검증과 연결해서 최종
   검증.

## 아직 안 한 것 (다음 사람이 이어서 할 일)

- [ ] `scene_right()` 함수 + `StereoSGBM` 통합
- [ ] `(R, p_eye)` 최소제곱 최적화 함수 작성 + 기존 `m` 모드와 별도 키로 통합
- [ ] 여러 거리 실측 데이터 수집 + 검증
- [ ] 기존 회전전용 `m` 모드는 "빠른 사전 정렬용"으로 남겨둘지, 완전히 교체할지 결정
