"""Train an AEGIS model from a YAML config.

Usage:
  python scripts/train.py configs/<name>.yaml
  python scripts/train.py --dump-teacher <checkpoint_dir> <config.yaml> <out.npz>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aegis.training.train import dump_teacher_logits, train  # noqa: E402

if __name__ == "__main__":
    if sys.argv[1] == "--dump-teacher":
        dump_teacher_logits(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        summary = train(sys.argv[1])
        print(json.dumps(summary["best"], indent=2))
