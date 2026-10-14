"""Token budget per tokenizer x normalization (input-side efficiency analysis).

Writes reports/token_budget.csv. Uses only the Function text (no labels).
Usage: python scripts/token_budget.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aegis.data.dataset import load_split  # noqa: E402
from aegis.data.preprocess import normalize  # noqa: E402
from aegis.paths import REPORTS_DIR  # noqa: E402

TOKENIZERS = ["microsoft/codebert-base", "microsoft/unixcoder-base"]
MODES = ["raw", "collapse", "single_line"]


def main() -> None:
    from transformers import AutoTokenizer

    rows = []
    for split in ("train", "test"):
        df = load_split(split)
        for name in TOKENIZERS:
            tok = AutoTokenizer.from_pretrained(name)
            for mode in MODES:
                texts = [normalize(s, mode) for s in df.Function]
                n = np.array([len(x) + 2 for x in tok(texts, add_special_tokens=False)["input_ids"]])
                clipped = np.minimum(n, 512)
                rows.append(dict(split=split, tokenizer=name, normalize=mode, mean=round(n.mean(), 1),
                                 median=float(np.median(n)), p95=float(np.percentile(n, 95)),
                                 frac_gt_512=round((n > 512).mean(), 4), mean_clipped_512=round(clipped.mean(), 1),
                                 total_tokens_clipped_512=int(clipped.sum()),
                                 official_padded_positions=int(len(n) * min(512, n.max()))))
                print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    out.to_csv(REPORTS_DIR / "token_budget.csv", index=False)


if __name__ == "__main__":
    main()
