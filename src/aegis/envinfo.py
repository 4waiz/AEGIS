"""Capture hardware/software provenance for experiment records."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys

from aegis.paths import DATASET_REVISION, REPO_ROOT


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO_ROOT) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def environment() -> dict:
    import numpy
    import torch
    import transformers

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit(),
        "dataset_revision": DATASET_REVISION,
        "disk_free_gb": round(shutil.disk_usage(REPO_ROOT).free / 1e9, 1),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=props.name,
            gpu_mem_gb=round(props.total_memory / 1e9, 1),
            gpu_capability=f"{props.major}.{props.minor}",
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
        )
        info["gpu_power_state"] = gpu_power_state()
        info["power_scheme"] = power_scheme()
    return info


def gpu_power_state() -> str:
    """Power draw/limit and clocks; runtime is only comparable under the same power mode."""
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw,power.max_limit,clocks.sm,clocks.max.sm,pstate",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        return "unknown"


def power_scheme() -> str:
    if platform.system() != "Windows":
        return "n/a"
    try:
        return subprocess.check_output(["powercfg", "/getactivescheme"], text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    print(json.dumps(environment(), indent=2))
