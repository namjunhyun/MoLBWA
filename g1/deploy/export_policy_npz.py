#!/usr/bin/env python3
"""체크포인트를 실기용 npz 로 빼고, 카탈로그에 설명을 자동 기록한다.

torch 없이 numpy 만으로 정책을 돌리기 위한 변환이다 (Jetson aarch64 에 torch 를
올리지 않는다. 이 정책은 MLP 4층이라 numpy 로 0.167 ms/회 = 제어주기의 0.8%).

메타는 손으로 적지 않는다. 학습 로그가 다 갖고 있다:
  params/env.yaml  →  지연 랜덤화 여부(class_type), 관측 구성(154/160), 짝 모션
  events.out.*     →  최종 보상, error_body_pos
정책이 늘어나도 카탈로그가 스스로 채워지므로 콘솔이 "무엇을 골라야 하는지" 알 수 있다.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CATALOG = HERE / "catalog.json"

AP = argparse.ArgumentParser()
AP.add_argument("--ckpt", required=True, help="model_*.pt (학습 로그 안의 것)")
AP.add_argument("--out", required=True, help="내보낼 npz 경로")
AP.add_argument("--title", default="", help="카드에 보일 이름. 없으면 파일명에서 만든다")
AP.add_argument("--note", default="", help="한 줄 설명. 없으면 메타로 자동 생성")
args = AP.parse_args()

ck = Path(args.ckpt).resolve()
run = ck.parent


def read_env_meta(run_dir):
    """env.yaml 은 python/tuple 태그가 있어 safe_load 가 안 된다 → 필요한 줄만 읽는다."""
    m = {"obs": None, "delay": False, "delay_steps": None, "motion": None, "task_hint": ""}
    p = run_dir / "params" / "env.yaml"
    if not p.exists():
        return m
    t = p.read_text(errors="ignore")
    m["delay"] = "DelayedJointPositionAction" in t
    d = re.search(r"max_delay_steps:\s*(\d+)", t)
    if d:
        m["delay_steps"] = int(d.group(1))
    # 실기용(Wo-State-Estimation) 은 이 두 관측이 null 로 꺼져 있다
    wose = "motion_anchor_pos_b: null" in t and "base_lin_vel: null" in t
    m["obs"] = 154 if wose else 160
    mo = re.search(r"motion_file:\s*(\S+)", t)
    if mo:
        m["motion"] = Path(mo.group(1)).name
    m["task_hint"] = ("Wo-State-Estimation" + ("-Delay" if m["delay"] else "")) if wose else "기본"
    return m


def read_scalars(run_dir):
    out = {}
    try:
        from tensorboard.backend.event_processing import event_accumulator as ea
        fs = glob.glob(str(run_dir / "events.out.tfevents.*"))
        if not fs:
            return out
        a = ea.EventAccumulator(fs[0], size_guidance={"scalars": 0}); a.Reload()
        for key, name in (("Train/mean_reward", "reward"),
                          ("Metrics/motion/error_body_pos", "body_err"),
                          ("Metrics/motion/error_joint_pos", "joint_err")):
            if key in a.Tags()["scalars"]:
                out[name] = round(float(a.Scalars(key)[-1].value), 4)
    except Exception:
        pass
    return out


# ── 가중치 ─────────────────────────────────────────────────────────────────
sd = torch.load(ck, map_location="cpu", weights_only=False)["actor_state_dict"]
d = {}
for i, k in enumerate((0, 2, 4, 6)):
    d[f"w{i}"] = sd[f"mlp.{k}.weight"].numpy().astype(np.float32)
    d[f"b{i}"] = sd[f"mlp.{k}.bias"].numpy().astype(np.float32)
d["obs_mean"] = sd["obs_normalizer._mean"].numpy().reshape(-1).astype(np.float32)
d["obs_std"] = sd["obs_normalizer._std"].numpy().reshape(-1).astype(np.float32)
# 학습은 (x - mean) / (std + eps) 로 정규화한다(rsl_rl EmpiricalNormalization, eps 기본 1e-2).
# 2026-10-08 전까지 이 eps 를 빼고 내보냈다 — 정지 자세처럼 std 가 0 인 차원이 있으면 배포에서 0 으로 나눠 NaN.
from rsl_rl.modules.normalization import EmpiricalNormalization
_norm = EmpiricalNormalization(d["obs_mean"].shape[0])
_norm._mean.copy_(sd["obs_normalizer._mean"].reshape(1, -1)); _norm._std.copy_(sd["obs_normalizer._std"].reshape(1, -1))
d["obs_eps"] = np.float32(_norm.eps)
out = Path(args.out).resolve()
np.savez(out, **d)

# torch 경로와 일치하는지 확인한다 — 이 검증 없이 실기에 올리지 않는다
def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
W = [d[f"w{i}"] for i in range(4)]; B = [d[f"b{i}"] for i in range(4)]
rng = np.random.default_rng(0); worst = 0.0
for _ in range(200):
    o = rng.standard_normal(d["obs_mean"].shape[0]).astype(np.float32)
    x = (o - d["obs_mean"]) / (d["obs_std"] + d["obs_eps"])
    for k in range(3):
        x = elu(W[k] @ x + B[k])
    a_np = W[3] @ x + B[3]
    with torch.no_grad():
        t = _norm.eval()(torch.from_numpy(o).reshape(1, -1)).reshape(-1)   # 라이브러리 공식 그대로 (자기참조 대조 금지)
        for k in (0, 2, 4):
            t = torch.nn.functional.elu(sd[f"mlp.{k}.weight"] @ t + sd[f"mlp.{k}.bias"])
        a_t = (sd["mlp.6.weight"] @ t + sd["mlp.6.bias"]).numpy()
    worst = max(worst, float(np.abs(a_np - a_t).max()))

# ── 카탈로그 ───────────────────────────────────────────────────────────────
em = read_env_meta(run)
sc = read_scalars(run)
it = int(ck.stem.split("_")[1]) if "_" in ck.stem else None
obs_dim = int(d["obs_mean"].shape[0])
if em["obs"] and em["obs"] != obs_dim:
    print(f"[주의] env.yaml 은 {em['obs']}차원인데 가중치는 {obs_dim}차원이다 — 가중치를 따른다")

note = args.note
if not note:
    bits = ["실기 가능" if obs_dim == 154 else "실기 불가(world 위치 요구)"]
    if em["delay"]:
        bits.append(f"지연 랜덤화 0~{(em['delay_steps'] or 0)*20}ms")
    if "reward" in sc:
        bits.append(f"보상 {sc['reward']:.1f}")
    if "body_err" in sc:
        bits.append(f"body {sc['body_err']:.4f}")
    note = " · ".join(bits)

entry = {
    "title": args.title or out.stem.replace("policy_", ""),
    "run": run.name, "iter": it, "obs": obs_dim,
    "delay": em["delay"], "delay_steps": em["delay_steps"],
    "motion": em["motion"], "task": em["task_hint"],
    "note": note, "export_diff": round(worst, 9),
}
entry.update(sc)

cat = {"policies": {}, "motions": {}}
if CATALOG.exists():
    try:
        cat = json.loads(CATALOG.read_text())
    except Exception:
        pass
cat.setdefault("policies", {})[out.name] = entry
CATALOG.write_text(json.dumps(cat, indent=1, ensure_ascii=False))

print(f"저장: {out.name}  {obs_dim}차원 · 파라미터 {sum(v.size for v in W)+sum(v.size for v in B):,}개")
print(f"torch 대비 최대 출력 차이: {worst:.2e}  ({'일치' if worst < 1e-4 else '불일치 — 확인 필요'})")
print(f"카탈로그 기록: {entry['title']}  |  {note}")
print(f"짝 모션: {em['motion'] or '(env.yaml 에서 못 읽음)'}")
