"""Cross-encoder pair scorer (GPU; runs as a SageMaker training job).

A multilingual transformer reads both records' raw name + address together and
predicts p(same business). Its score is meant as one more feature for the LightGBM
matcher: it sees the raw text, which the hand-built similarity features only
summarise. xlm-roberta-base is MIT-licensed (~280M params) and pretrained on French
and the Indic scripts, which matters for countries we hold no labels for.

Channels (SageMaker mounts each under /opt/ml/input/data/<name>/):
    train   parquet with a_name, a_addr, b_name, b_addr, label        (fine-tuning)
    eval    same columns; scored after training and reported (AUC / logloss)
    score   optional: pairs to score without labels (e.g. the test candidates)
    model   optional: a previous job's model.tar.gz to start from (scoring-only runs)
Scores are written as parquet to the output-data dir; the fine-tuned model to /opt/ml/model.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)


def read_channel(name):
    d = os.environ.get(f"SM_CHANNEL_{name.upper()}", f"/opt/ml/input/data/{name}")
    files = sorted(glob.glob(os.path.join(d, "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def side(name, addr):
    return (name.fillna("") + " | " + addr.fillna("")).tolist()


class Collate:
    def __init__(self, tok, max_len):
        self.tok, self.max_len = tok, max_len

    def __call__(self, batch):
        a, b, y = zip(*batch)
        enc = self.tok(list(a), list(b), truncation="longest_first", max_length=self.max_len,
                       padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc


def loader(df, tok, args, bs, shuffle):
    y = df["label"].tolist() if "label" in df else [0.0] * len(df)
    rows = list(zip(side(df["a_name"], df["a_addr"]), side(df["b_name"], df["b_addr"]), y))
    return DataLoader(rows, batch_size=bs, shuffle=shuffle, num_workers=args.workers,
                      collate_fn=Collate(tok, args.max_len), pin_memory=True)


@torch.no_grad()
def predict(model, df, tok, args, dev):
    """Scores in df's row order. Batches are built over length-sorted rows so each
    batch pads to similar lengths -- ~2x faster than padding random mixes."""
    model.eval()
    order = np.argsort((df["a_name"].str.len() + df["a_addr"].str.len()
                        + df["b_name"].str.len() + df["b_addr"].str.len()).to_numpy())
    out = []
    for enc in loader(df.iloc[order], tok, args, args.eval_bs, False):
        enc.pop("labels")
        enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**enc).logits.float().squeeze(-1).cpu())
    p = np.empty(len(df), dtype=np.float32)
    p[order] = torch.sigmoid(torch.cat(out)).numpy()
    return p


def report(p, y, tag):
    from sklearn.metrics import log_loss, roc_auc_score
    print(f"[{tag}] n={len(y):,} AUC={roc_auc_score(y, p):.5f} "
          f"logloss={log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)):.5f} "
          f"acc@0.5={((p >= 0.5) == y).mean():.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--eval_bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--save_every", type=int, default=0, help="checkpoint every N steps")
    args, _ = ap.parse_known_args()
    out_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)

    train, evald, score = read_channel("train"), read_channel("eval"), read_channel("score")
    init = os.environ.get("SM_CHANNEL_MODEL")
    if init and os.path.exists(os.path.join(init, "model.tar.gz")):
        import tarfile
        with tarfile.open(os.path.join(init, "model.tar.gz")) as t:
            t.extractall("/tmp/init_model")
        args.model = "/tmp/init_model"
        print("starting from a previously fine-tuned model", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(dev)

    if train is not None:
        print(f"train {len(train):,} pairs ({train['label'].mean():.1%} true)", flush=True)
        dl = loader(train, tok, args, args.bs, True)
        steps = math.ceil(len(dl) * args.epochs)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
        lossf = torch.nn.BCEWithLogitsLoss()
        model.train()
        step, t0, run = 0, time.time(), 0.0
        while step < steps:
            for enc in dl:
                y = enc.pop("labels").to(dev, non_blocking=True)
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = model(**enc).logits.float().squeeze(-1)
                loss = lossf(logit, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
                step += 1
                if step % 500 == 0:
                    rate = step * args.bs / (time.time() - t0)
                    print(f"step {step}/{steps} loss {run:.4f} {rate:,.0f} pairs/s", flush=True)
                if args.save_every and step % args.save_every == 0:
                    model.save_pretrained(model_dir)       # crash insurance on long runs
                    tok.save_pretrained(model_dir)
                if step >= steps:
                    break
        model.save_pretrained(model_dir)
        tok.save_pretrained(model_dir)
        del train, dl                                     # free RAM before scoring

    for name, df in (("eval", evald), ("score", score)):
        if df is None:
            continue
        t = time.time()
        p = predict(model, df, tok, args, dev)
        print(f"scored {name}: {len(df):,} pairs in {time.time() - t:.0f}s", flush=True)
        if "label" in df:
            report(p, df["label"].to_numpy(), name)
        keep = [c for c in ("s1_entity_id", "cand_entity_id", "label") if c in df]
        df[keep].assign(ce_p=p.astype(np.float32)).to_parquet(
            os.path.join(out_dir, f"{name}_ce.parquet"), index=False)


if __name__ == "__main__":
    main()
