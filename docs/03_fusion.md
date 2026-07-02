# 03. 융합 — 이 프로젝트의 심장

가장 먼저 뚫어야 할 리스크. **책상 위에서 오프라인으로 먼저 검증**한다.

## 변환 사슬 → 코드
```python
import numpy as np

def gaze_point_world(u, v, D, K, T_WS):
    """
    u, v : 캘리브레이션 매핑으로 얻은 스테레오 이미지상의 시선 픽셀
    D    : 그 픽셀의 스테레오 깊이 (StereoSGBM, 시선 주변 윈도우)
    K    : 스테레오(rectified) 카메라 내부 파라미터 3x3
    T_WS : 해당 타임스탬프의 SLAM pose (4x4, world <- sensor)
    반환 : (p_W 3D점, ray_origin, ray_dir)
    """
    # 1) 픽셀 → 카메라 좌표 3D 역투영
    Kinv = np.linalg.inv(K)
    p_S = D * (Kinv @ np.array([u, v, 1.0]))        # 카메라 좌표계 3D점

    # 2) 카메라 → 월드
    p_S_h = np.array([p_S[0], p_S[1], p_S[2], 1.0])
    p_W = (T_WS @ p_S_h)[:3]

    # 3) 시각화용 시선 광선
    origin = T_WS[:3, 3]                             # 카메라 중심 (pose translation)
    ray_dir = p_W - origin
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    return p_W, origin, ray_dir
```

## 단계별 의미
1. **시선 픽셀** `(u,v)`: 캘리브레이션 매핑으로 스테레오 이미지 위에서 구한다.
2. **역투영**: `p_S = D · K⁻¹ [u,v,1]ᵀ` — 스테레오 깊이 `D`로 카메라 좌표 3D점.
3. **월드 변환**: `p_W = T_WS · p_S`.
4. **시선 광선**: 원점 = 카메라 중심, 방향 = `normalize(p_W − 원점)`.

## "얼마나 떨어졌나?"
= 그냥 **`D`(깊이)**.
눈과 스테레오 카메라가 머리 위 몇 cm 차이라, 눈→대상 거리와 카메라→대상 거리는
**실용적으로 같다**고 봐도 된다.

## 검증 방법 (0단계)
- 알려진 거리에 물체를 두고 응시 → `D`, `p_W`가 실제와 맞는지 자로 확인.
- pose를 흔들며 같은 물체 응시 → `p_W`가 월드에서 고정되는지 확인 (융합이 맞으면 고정).

## 체크리스트
- [ ] `gaze_point_world` 단위 테스트 (합성 데이터)
- [ ] 실물 물체로 D/p_W 정확도 확인
- [ ] pose 이동 시 p_W 월드 고정성 확인
