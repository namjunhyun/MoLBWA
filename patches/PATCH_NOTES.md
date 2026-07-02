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
