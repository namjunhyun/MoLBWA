"""모션 서버 스모크: 인자 파싱과 154차원 가드. 로봇·DDS·MuJoCo 를 열지 않는다
(가드는 PID 파일·소켓·백엔드보다 먼저 돈다)."""
import os
import subprocess
import sys

import numpy as np

SERVER = os.path.join(os.path.dirname(__file__), "..", "deploy", "g1_motion_server.py")


def run(*a):
    return subprocess.run([sys.executable, SERVER, *a], capture_output=True, text=True, timeout=60)


def test_parses_args():
    r = run("--help")
    assert r.returncode == 0, r.stderr
    assert "--library_meta" in r.stdout and "--fake_events" in r.stdout


def test_obs_dim_guard(tmp_path):
    pol = tmp_path / "policy160.npz"
    np.savez(pol, **{f"w{i}": np.zeros((2, 2)) for i in range(4)},
             **{f"b{i}": np.zeros(2) for i in range(4)},
             obs_mean=np.zeros(160), obs_std=np.ones(160))
    r = run("--policy", str(pol), "--motion", str(tmp_path / "none.npz"),
            "--library_meta", str(tmp_path / "none.json"))
    assert r.returncode != 0
    assert "154" in r.stderr + r.stdout
