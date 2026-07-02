#!/usr/bin/env python3
"""
클로즈업 눈 영상(eye_test.mp4)에 Orlosky 3D 트래커를 돌려
'동공이 제대로 잡히는가 + 시선 벡터가 의미있게 나오는가'를 검증.
(RealSense는 눈을 못 채워서 검출 품질 검증 불가 → 실제 눈 영상으로 알고리즘만 확인)

헤드리스: 주요 프레임을 오버레이와 함께 data/eye_XXXX.png로 저장 + 벡터 통계 출력.
"""
import os, sys, argparse
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKER_DIR = os.path.join(HERE, "..", "external", "EyeTracker", "3DTracker")
sys.path.insert(0, os.path.abspath(TRACKER_DIR))
os.chdir(HERE)
GAZE_TXT = os.path.join(HERE, "gaze_vector.txt")
if os.path.exists(GAZE_TXT):
    os.remove(GAZE_TXT)
import Orlosky3DEyeTracker as tracker  # noqa

def read_last_gaze():
    if not os.path.exists(GAZE_TXT): return None
    try:
        with open(GAZE_TXT) as f:
            lines=[l.strip() for l in f if l.strip()]
        vals=[float(x) for x in lines[-1].replace(",", " ").split()]
        if len(vals)>=6: return np.array(vals[:3]), np.array(vals[3:6])
    except Exception: return None
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(HERE,"..","external","EyeTracker","eye_test.mp4"))
    ap.add_argument("--save-every", type=int, default=40)
    args=ap.parse_args()

    outdir=os.path.join(HERE,"..","data"); os.makedirs(outdir, exist_ok=True)
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("영상 못 엶:", args.video); return

    n=0; saved=0; dirs=[]; mcs=[]
    while True:
        ret, frame = cap.read()
        if not ret: break
        rr = tracker.process_frame(frame)   # 검출+모델+벡터+저장, 그림은 frame 내부로 그려짐
        if rr is None:
            rr = ((0, 0), (0, 0), 0)
        n+=1
        g=read_last_gaze(); mc=len(getattr(tracker,"model_centers",[]))
        mcs.append(mc)
        if g is not None:
            dirs.append(g[1])
        # 모델이 선 이후 구간 위주로 저장
        if n % args.save_every == 0:
            # process_frame이 그린 창 프레임을 다시 못 받으므로, 원본에 검출 타원만 재오버레이
            vis=frame.copy()
            (cx,cy),(w,h),ang = rr
            if w>0 and h>0:
                cv2.ellipse(vis, rr, (0,255,0), 2)
                cv2.circle(vis,(int(cx),int(cy)),3,(0,255,255),-1)
            path=os.path.join(outdir, f"eye_{n:04d}.png")
            cv2.imwrite(path, vis); saved+=1
            dd = dirs[-1] if dirs else np.zeros(3)
            print(f"frame {n:4d}  model_centers={mc:3d}  pupil={'yes' if w>0 else 'no'}  "
                  f"dir=({dd[0]:+.3f},{dd[1]:+.3f},{dd[2]:+.3f})  -> {os.path.basename(path)}")
    cap.release(); cv2.destroyAllWindows()

    d=np.array(dirs)
    print(f"\n[요약] 총 {n}프레임, 벡터 {len(d)}개, 최종 model_centers={mcs[-1] if mcs else 0}")
    if len(d)>30:
        tail=d[-60:]
        print("  마지막 60프레임 dir 평균:", np.round(tail.mean(0),3),
              " 표준편차:", np.round(tail.std(0),3))
    print(f"  저장 이미지 {saved}장 → data/, 벡터로그 → {GAZE_TXT}")

if __name__=="__main__":
    main()
