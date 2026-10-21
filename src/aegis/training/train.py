"""Training loop for AEGIS models (multitask BCE, optional multitask distillation).

The official public test split is never touched here: model selection and threshold tuning use
the internal validation split carved out of the official training split.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from aegis.data.dataset import load_internal_splits
from aegis.data.preprocess import build_inputs, normalize, tokenize_cached
from aegis.envinfo import environment
from aegis.evaluation.metrics import f1_at, tune_thresholds
from aegis.models.multitask import build_model, count_parameters
from aegis.paths import ARTIFACTS_DIR, EXPERIMENTS_DIR


def load_config(path: str) -> dict:
    """Load a YAML config; string values may use environment variables, e.g. ${AEGIS_ARTIFACTS}."""
    import os

    def expand(v):
        if isinstance(v, dict):
            return {k: expand(x) for k, x in v.items()}
        if isinstance(v, list):
            return [expand(x) for x in v]
        return os.path.expandvars(v) if isinstance(v, str) else v

    os.environ.setdefault("AEGIS_ARTIFACTS", str(ARTIFACTS_DIR))
    return expand(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def length_grouped_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator,
                           mega: int = 64) -> list[np.ndarray]:
    """Shuffle, then sort inside mega-batches by length so padding stays small."""
    idx = rng.permutation(len(lengths))
    batches = []
    for start in range(0, len(idx), batch_size * mega):
        chunk = idx[start : start + batch_size * mega]
        chunk = chunk[np.argsort(-lengths[chunk], kind="stable")]
        batches.extend(chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size))
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


def collate(seqs: list[np.ndarray], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, : len(s)] = torch.from_numpy(s)
        mask[r, : len(s)] = 1
    return ids, mask


def prepare_split(df, tokenizer, data_cfg: dict, tag: str, mode: str | None = None) -> list[np.ndarray]:
    """Normalize, tokenize (cached) and truncate. mode="mixed" (training only) normalizes a fixed,
    seeded half of the functions as single_line and leaves the other half raw, so that the model
    sees both input formats."""
    mode = mode or data_cfg.get("normalize", "raw")
    if mode == "mixed":
        pick = np.random.default_rng(0).random(len(df)) < 0.5
        texts = [normalize(s, "single_line" if p else "raw") for s, p in zip(df["Function"].tolist(), pick)]
    else:
        texts = [normalize(s, mode) for s in df["Function"].tolist()]
    cache = tokenize_cached(texts, tokenizer, f"{tag}_{mode}")
    prefix = tokenizer.convert_tokens_to_ids(data_cfg["prefix_tokens"]) if data_cfg.get("prefix_tokens") else None
    return build_inputs(cache, tokenizer.cls_token_id, tokenizer.sep_token_id,
                        data_cfg.get("max_length", 512), prefix, data_cfg.get("truncation", "head"))


@torch.no_grad()
def predict_logits(model, inputs: list[np.ndarray], pad_id: int, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    order = np.argsort([-len(x) for x in inputs], kind="stable")
    out = np.zeros((len(inputs), model.config.num_labels), dtype=np.float32)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        ids, mask = collate([inputs[i] for i in idx], pad_id)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        out[idx] = logits.float().cpu().numpy()
    return out


def multitask_loss(logits: torch.Tensor, labels: torch.Tensor, cfg: dict,
                   teacher: torch.Tensor | None = None) -> torch.Tensor:
    """Per-task BCE (optionally focal / positive-weighted), averaged with task weights, plus
    optional per-task logit distillation with temperature T (Hinton et al., 2015)."""
    logits = logits.float()
    pos_weight = torch.tensor(cfg.get("pos_weight", [1.0, 1.0]), device=logits.device)
    task_w = torch.tensor(cfg.get("task_weights", [1.0, 1.0]), device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    gamma = cfg.get("focal_gamma", 0.0)
    if gamma > 0:
        p = torch.sigmoid(logits)
        pt = labels * p + (1 - labels) * (1 - p)
        bce = bce * (1 - pt).pow(gamma)
    hard = (bce.mean(0) * task_w).sum() / task_w.sum()
    if teacher is None:
        return hard
    T = cfg.get("kd_temperature", 2.0)
    alpha = cfg.get("kd_alpha", 0.5)
    soft_t = torch.sigmoid(teacher.float() / T)
    kd = F.binary_cross_entropy_with_logits(logits / T, soft_t, reduction="none") * T * T
    kd = (kd.mean(0) * task_w).sum() / task_w.sum()
    return alpha * hard + (1 - alpha) * kd


def train(config_path: str) -> dict:
    cfg = load_config(config_path)
    name = cfg["name"]
    tcfg, dcfg, mcfg = cfg["train"], cfg["data"], cfg["model"]
    exp_dir = EXPERIMENTS_DIR / name
    ckpt_dir = ARTIFACTS_DIR / name / "best"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    env = environment()
    (exp_dir / "env.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")

    set_seed(tcfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    splits = load_internal_splits()
    if tcfg.get("subsample"):  # smoke tests only
        splits["fit"] = splits["fit"].sample(tcfg["subsample"], random_state=0)
        splits["val"] = splits["val"].sample(min(len(splits["val"]), tcfg["subsample"] // 4), random_state=0)
    model, tokenizer = build_model(mcfg)
    model.to(device)
    pad_id = tokenizer.pad_token_id
    tok_tag = mcfg.get("tokenizer") or mcfg["init"]
    if mcfg["init"] == "scratch":
        tok_tag = mcfg["tokenizer"]
    tok_tag = tok_tag.replace("/", "_")
    fit_inputs = prepare_split(splits["fit"], tokenizer, dcfg, f"fit_{tok_tag}", mode=dcfg.get("train_normalize"))
    val_inputs = prepare_split(splits["val"], tokenizer, dcfg, f"val_{tok_tag}")
    y_fit = splits["fit"][["y_vuln", "y_mat"]].to_numpy().astype(np.float32)
    y_val = splits["val"][["y_vuln", "y_mat"]].to_numpy()

    teacher_logits = None
    if cfg.get("distill", {}).get("teacher_logits"):
        z = np.load(cfg["distill"]["teacher_logits"])
        lookup = dict(zip(z["row_id"].tolist(), range(len(z["row_id"]))))
        teacher_logits = z["logits"][[lookup[r] for r in splits["fit"]["row_id"].tolist()]]

    lengths = np.array([len(x) for x in fit_inputs])
    bs = tcfg.get("batch_size", 32)
    accum = tcfg.get("grad_accum", 1)
    epochs = tcfg.get("epochs", 3)
    steps_per_epoch = math.ceil(len(fit_inputs) / bs / accum)
    total_steps = steps_per_epoch * epochs
    no_decay = ["bias", "LayerNorm.weight", "layer_norm"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": tcfg.get("weight_decay", 0.01)},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), betas=(0.9, 0.999), eps=1e-8,
                            fused=device.type == "cuda")
    warmup = int(total_steps * tcfg.get("warmup_ratio", 0.06))

    def lr_lambda(s: int) -> float:  # linear warmup, then linear decay to 0
        if s < warmup:
            return (s + 1) / max(1, warmup)
        return max(0.0, (total_steps - s) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    rng = np.random.default_rng(tcfg.get("seed", 42))
    log_path = exp_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    best = {"val_mean_f1_tuned": -1.0}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    step = 0
    patience = tcfg.get("patience", 2)
    bad_epochs = 0
    eval_every = tcfg.get("eval_every_epochs", 1.0)
    for epoch in range(epochs):
        best_before = best["val_mean_f1_tuned"]
        model.train()
        batches = length_grouped_batches(lengths, bs, rng)
        n_eval_points = max(1, round(1 / eval_every)) if eval_every < 1 else 1
        eval_at = {round(len(batches) * (k + 1) / n_eval_points) - 1 for k in range(n_eval_points)}
        running, n_run = 0.0, 0
        for b_i, idx in enumerate(batches):
            ids, mask = collate([fit_inputs[i] for i in idx], pad_id)
            labels = torch.from_numpy(y_fit[idx]).to(device)
            teach = torch.from_numpy(teacher_logits[idx]).to(device) if teacher_logits is not None else None
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=tcfg.get("bf16", True)):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = multitask_loss(logits, labels, tcfg, teach) / accum
            loss.backward()
            running += loss.item() * accum
            n_run += 1
            if (b_i + 1) % accum == 0 or b_i == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("max_grad_norm", 1.0))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % tcfg.get("log_every_steps", 500) == 0:
                    print(f"step {step}/{total_steps} loss {running / max(1, n_run):.4f} "
                          f"{(time.time() - t_start) / 60:.1f} min", flush=True)
            if b_i in eval_at:
                val_logits = predict_logits(model, val_inputs, pad_id, device)
                probs = 1 / (1 + np.exp(-val_logits))
                at05 = f1_at(probs, y_val, [0.5, 0.5])
                thr = tune_thresholds(probs, y_val)
                tuned = f1_at(probs, y_val, thr)
                rec = {"epoch": round(epoch + (b_i + 1) / len(batches), 3), "step": step,
                       "train_loss": running / max(1, n_run), "elapsed_s": round(time.time() - t_start, 1),
                       "val_at_0.5": at05, "val_tuned": tuned, "thresholds": thr}
                running, n_run = 0.0, 0
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps({k: rec[k] for k in ("epoch", "train_loss", "elapsed_s")}),
                      f"val@0.5 {at05['mean_f1']:.4f} (V {at05['f1_vuln']:.4f} M {at05['f1_mat']:.4f})",
                      f"tuned {tuned['mean_f1']:.4f} thr {thr}", flush=True)
                if tuned["mean_f1"] > best["val_mean_f1_tuned"]:
                    best = {"val_mean_f1_tuned": tuned["mean_f1"], "val_tuned": tuned, "val_at_0.5": at05,
                            "thresholds": thr, "epoch": rec["epoch"]}
                    model.config.aegis = {"thresholds": thr, "normalize": dcfg.get("normalize", "raw"),
                                          "max_length": dcfg.get("max_length", 512),
                                          "truncation": dcfg.get("truncation", "head"),
                                          "prefix_tokens": dcfg.get("prefix_tokens")}
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    np.savez(exp_dir / "val_logits_best.npz", logits=val_logits,
                             row_id=splits["val"]["row_id"].to_numpy())
                model.train()
        bad_epochs = 0 if best["val_mean_f1_tuned"] > best_before else bad_epochs + 1
        if bad_epochs >= patience:
            print(f"early stop after epoch {epoch + 1}")
            break

    summary = {
        "name": name, "config": cfg, "best": best, "checkpoint": str(ckpt_dir),
        "train_time_s": round(time.time() - t_start, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "params": count_parameters(model), "n_fit": len(fit_inputs), "n_val": len(val_inputs),
        "mean_fit_tokens": float(lengths.mean()), "env": env,
    }
    (exp_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def dump_teacher_logits(checkpoint: str, config_path: str, out: str) -> None:
    """Teacher logits on the internal fit split, keyed by row_id, for distillation."""
    cfg = load_config(config_path)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    splits = load_internal_splits()
    tok_tag = (cfg["model"].get("tokenizer") or cfg["model"]["init"]).replace("/", "_")
    inputs = prepare_split(splits["fit"], tokenizer, cfg["data"], f"fit_{tok_tag}")
    logits = predict_logits(model, inputs, tokenizer.pad_token_id, device)
    np.savez(out, logits=logits, row_id=splits["fit"]["row_id"].to_numpy())
    print("saved", out, logits.shape)
