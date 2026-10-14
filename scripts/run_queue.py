"""Run several configs sequentially: train, then score on the INTERNAL VALIDATION split with the
official measurement loop (selection never looks at public-test labels).

Usage: python scripts/run_queue.py configs/a.yaml configs/b.yaml ...
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("AEGIS_ARTIFACTS", ROOT / "artifacts"))


def main() -> None:
    for cfg_path in sys.argv[1:]:
        name = yaml.safe_load(Path(cfg_path).read_text())["name"]
        exp = ROOT / "experiments" / name
        exp.mkdir(parents=True, exist_ok=True)
        print(f"=== {name}", flush=True)
        with open(exp / "train.log", "w") as log:
            rc = subprocess.call([sys.executable, "-W", "ignore", "scripts/train.py", cfg_path], cwd=ROOT,
                                 stdout=log, stderr=subprocess.STDOUT)
        if rc != 0:
            print(f"training failed for {name} (see {exp / 'train.log'})", flush=True)
            continue
        with open(exp / "measure_val.log", "w") as log:
            subprocess.call([sys.executable, "-W", "ignore", "scripts/measure.py", str(ART / name / "best"),
                             "--split", "val", "--out", str(exp / "measure_val.json")], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT)
        print(f"done {name}", flush=True)


if __name__ == "__main__":
    main()
