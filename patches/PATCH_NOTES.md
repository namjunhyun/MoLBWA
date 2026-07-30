# 외부 레포 패치 노트 — numpy 2.x uint8 overflow 수정

대상: [JEOresearch/EyeTracker](https://github.com/JEOresearch/EyeTracker) (Orlosky 동공/시선 검출기)

## 왜 필요한가
`get_darkest_area()`와 `apply_binary_threshold()`가 **uint8 픽셀끼리 덧셈**을 해서
numpy 2.x에서 **overflow(wrap-around)** 가 난다. 임계값이 엉켜 동공이 아니라
눈꺼풀 개구부 전체를 잡는 오검출이 발생한다. (레포 README도 numpy 2.0 이슈를 경고함.)

## 수정 (각 파일 2곳, `int()` 캐스팅 추가)

### `OrloskyPupilDetectorLite.py`
```python
# apply_binary_threshold()
- threshold = darkestPixelValue + addedThreshold
+ threshold = int(darkestPixelValue) + addedThreshold

# get_darkest_area()
- current_sum += gray[y + dy][x + dx]
+ current_sum += int(gray[y + dy][x + dx])
```

### `3DTracker/Orlosky3DEyeTracker.py`  (같은 두 함수, 대략 line 66 / 91)
```python
- threshold = darkestPixelValue + addedThreshold
+ threshold = int(darkestPixelValue) + addedThreshold

- current_sum += gray[y + dy][x + dx]
+ current_sum += int(gray[y + dy][x + dx])
```

## 효과
수정 후 동공에 타원 정확히 락온, 안구 구 모델 수렴(model_centers→200),
3D 시선 벡터가 실제 시선과 일치.

> 참고: 원본 파일은 CRLF(윈도우) 줄바꿈이라 raw `.patch`는 줄바꿈 노이즈가 많다.
> 위 4줄만 손으로 바꾸면 된다.

---

# 패치 2 — 파일 IPC 제거, 시선벡터를 반환값으로 노출 (2026-07-30)

대상: `3DTracker/Orlosky3DEyeTracker.py`

## 왜 필요한가
`docs/09_visualization.md` 선행조건 1번. 기존엔 매 프레임 `gaze_vector.txt`에 썼다가
`gaze_on_scene.py`의 `read_last_gaze()`가 다시 읽는 구조였다. 데모에선 문제없지만,
pose/스테레오 깊이와 타임스탬프를 맞춰야 하는 융합(docs/03) 단계에서는 레이스 컨디션과
프레임 드랍 위험이 있어 미리 제거.

## 수정 (3곳)

### 1. `compute_gaze_vector()` — 파일 쓰기 블록 삭제
`file_path = "gaze_vector.txt"`로 시작하는 블록(`is_file_available`, `open(file_path, "w")` 등)
통째로 제거. `return sphere_center, gaze_rotated`만 남김 (반환값 자체는 그대로).

### 2. `process_frames()` — `last_tracking_result`에 시선벡터 추가
`center, direction = compute_gaze_vector(...)` 호출 바로 다음 줄에 추가:
```python
last_tracking_result["gaze_origin"] = center.tolist() if center is not None else None
last_tracking_result["gaze_direction"] = direction.tolist() if direction is not None else None
```
(`last_tracking_result`는 이미 `global` 선언되어 있어서 그대로 갱신됨)

### 3. `process_frame()` — 반환값에 시선벡터 포함
```python
def process_frame(frame):
    ...
    final_rotated_rect = process_frames(...)

    result = get_last_tracking_result()
    gaze_direction = None
    if result is not None and result.get("gaze_direction") is not None:
        gaze_direction = np.array(result["gaze_direction"], dtype=np.float32)

    return final_rotated_rect, gaze_direction   # 예전엔 final_rotated_rect만 반환
```

## 호출부 변경
`gaze_on_scene.py`도 같이 수정: `read_last_gaze()` 함수 삭제, `GAZE_TXT` 관련 코드 삭제,
`tracker.process_frame(eye)` 호출을 `_ellipse, d = tracker.process_frame(eye)`로 변경.

## 효과
파일 I/O·레이스 컨디션 없이 같은 프레임 안에서 시선벡터를 바로 받음. 동작(화면 표시, 다점
캘리브 등)은 기존과 동일 — 배선만 교체.
