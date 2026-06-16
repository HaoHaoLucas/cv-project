#!/usr/bin/env python3
"""尝试检测 / 编译 GroundingDINO CUDA 自定义算子。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GDINO = ROOT / "third_party" / "GroundingDINO"
OUT = ROOT / "results" / "exp_2026-06-15_cuda_built"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "build_attempt.log"


def log(msg: str) -> None:
    print(msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def try_import_c() -> bool:
    try:
        import groundingdino._C as _C  # noqa: F401

        log("OK: groundingdino._C import succeeded")
        return True
    except Exception as e:
        log(f"FAIL: groundingdino._C import: {e}")
        return False


def main() -> int:
    LOG.write_text("", encoding="utf-8")
    log("=== CUDA ops track ===")

    if try_import_c():
        (OUT / "status.json").write_text('{"status":"available"}\n', encoding="utf-8")
        return 0

    env = os.environ.copy()
    # CUDA 11.5 + gcc-11 会编译失败；优先 gcc-10
    for gcc in ("gcc-10", "gcc-11"):
        p = f"/usr/bin/{gcc}"
        if Path(p).exists():
            env["CC"] = p
            env["CXX"] = p.replace("gcc", "g++")
            log(f"Using compiler: {env['CC']}")
            break

    cuda_home = env.get("CUDA_HOME") or "/usr/local/cuda"
    if Path(cuda_home).exists():
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = f"{cuda_home}/bin:" + env.get("PATH", "")
        log(f"CUDA_HOME={cuda_home}")

    cmd = [sys.executable, "setup.py", "build", "develop"]
    log("Running: " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(GDINO),
        env=env,
        capture_output=True,
        text=True,
    )
    log(proc.stdout[-4000:] if proc.stdout else "")
    log(proc.stderr[-4000:] if proc.stderr else "")
    log(f"exit_code={proc.returncode}")

    ok = try_import_c() if proc.returncode == 0 else False
    status = "built" if ok else "fallback_pytorch"
    (OUT / "status.json").write_text(
        f'{{"status":"{status}","exit_code":{proc.returncode}}}\n',
        encoding="utf-8",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
