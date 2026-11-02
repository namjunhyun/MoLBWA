"""서버 sim 시연 영상에 단계 자막(대기 → 회전 → 동작)을 넣고 종료 절차 전까지 자른다.
    python3 caption_demo.py IN.mp4 IN.csv OUT.mp4 "제목 한 줄"
IN.csv 는 서버 --log_csv (phase 열 = 세그먼트 이름).
"""
import csv, subprocess, sys
src, log, out, title = sys.argv[1:5]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
LAB = {"wave": "손 흔들기", "handshake": "악수", "open_arms": "팔 벌리기"}
rows = list(csv.DictReader(open(log)))
spans = []
for r in rows:
    if not spans or spans[-1][0] != r["phase"]:
        spans.append([r["phase"], float(r["t"]), float(r["t"])])
    spans[-1][2] = float(r["t"])
for i in range(len(spans) - 1):
    spans[i][2] = spans[i + 1][1]
spans[-1][2] += 30


def esc(s):
    return s.replace(":", "\\:").replace("'", "’").replace("%", "\\%")


f = ["scale=640:-2", "drawbox=x=0:y=0:w=iw:h=34:color=black@0.55:t=fill",
     f"drawtext=fontfile={FONT}:text='{esc(title)}':x=12:y=9:fontsize=14:fontcolor=white",
     "drawbox=x=0:y=ih-44:w=iw:h=44:color=black@0.55:t=fill"]
seen = False
end = None
for ph, a, b in spans:
    if ph.startswith("turn"):
        t = "보행식 제자리 회전 (" + ph.replace("turn_l", "왼쪽 ").replace("turn_r", "오른쪽 ") + "°)"; seen = True
    elif ph == "idle":
        t = "대기" if seen else "대기 · 착용자가 응시하는 중"
    elif ph in LAB:
        t = LAB[ph]; seen = True
    else:
        end = end if end is not None else a
        continue
    col = "white" if ph == "idle" else "0xffd479"
    f.append(f"drawtext=fontfile={FONT}:text='{esc(t)}':x=12:y=h-33:fontsize=19:fontcolor={col}:enable='between(t,{a:.2f},{b - 0.04:.2f})'")
cut = ["-t", f"{end + 0.5:.2f}"] if end else []
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *cut, "-i", src, "-vf", ",".join(f), "-c:v", "libx264", "-crf", "28",
                "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", out], check=True)
print(out, [(s[0], round(s[1], 1)) for s in spans[:6]], "cut", end)
