#!/usr/bin/env python3
"""탑다운 호모그래피(640x480 YUYV) -> 1280x720 MJPG 로 환산 (2026-10-22). 팔을 다시 찍지 않는다.

이 카메라(0bda:5844)의 640x480 은 1280x720 가운데 960x720 을 2/3 로 줄인 것과 같다 (같은 태그
28 코너로 실측: u640 = s*u720 + tx, v640 = s*v720 + ty, 잔차 평균 0.12px).
그래서 H720 = H640 @ S, S = [[s,0,tx],[0,s,ty],[0,0,1]]. K, P, 캘리브 점 픽셀도 같이 바꾼다.
결과 파일에 image_size: [1280, 720] 를 적는다 — control_panel 이 실제 프레임 크기와 비교한다.

    python3 tools/topdown_to_720p.py            # 팔 a, b (원본은 *_bak_640x480_<시각>.yaml)
"""
import argparse
import os
import shutil
import time

import numpy as np
import yaml

S_FIT = (0.66695, -106.028, -0.040)   # 2026-10-22 실측 (이론값 0.66667, -106.667, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["a", "b"])
    ap.add_argument("--s", type=float, default=S_FIT[0])
    ap.add_argument("--tx", type=float, default=S_FIT[1])
    ap.add_argument("--ty", type=float, default=S_FIT[2])
    args = ap.parse_args()

    S = np.array([[args.s, 0, args.tx], [0, args.s, args.ty], [0, 0, 1.0]])
    Sinv = np.linalg.inv(S)
    for arm in args.arms:
        path = os.path.expanduser(f"~/.ros/topdown_homography_{arm}.yaml")
        d = yaml.safe_load(open(path))
        if d.get("image_size") == [1280, 720]:
            print(f"[건너뜀] {path}: 이미 1280x720")
            continue
        bak = path.replace(".yaml", time.strftime("_bak_640x480_%Y%m%d_%H%M%S.yaml"))
        shutil.copy2(path, bak)

        H = np.array(d["H"], float) @ S
        d["H"] = (H / H[2, 2]).tolist()
        for k in ("pixels", "pixels3d"):
            if d.get(k):
                px = np.array(d[k], float)
                d[k] = np.round((px - [args.tx, args.ty]) / args.s, 2).tolist()
        for k in ("K", "P"):
            if d.get(k) is not None:
                d[k] = (Sinv @ np.array(d[k], float)).tolist()
        d["image_size"] = [1280, 720]
        d["converted_from"] = {"image_size": [640, 480], "s": args.s, "tx": args.tx, "ty": args.ty,
                               "backup": bak, "stamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(path, "w") as f:
            yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True)
        print(f"[변환] {path}  (원본 {bak})")


if __name__ == "__main__":
    main()
