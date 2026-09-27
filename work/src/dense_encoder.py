"""Dense retrieval channel for blocking (GPU; runs as a SageMaker training job).

1. Fine-tune a multilingual bi-encoder (multilingual-e5-small, MIT, 118M params) with
   in-batch negatives on ground-truth pairs of S1 entities OUTSIDE validation fold 0,
   one match per S1 per pass so an entity never appears twice in a batch (which would
   make its own match a false negative).
2. Embed every record of both splits (raw name | address, so no cleaning is assumed).
3. Exact nearest-neighbour search on the GPU, per split and country:
     dense_<split>_<country>.parquet   each S2/S3 record's top-k S1 records
     links_<split>_<country>.parquet   each S2 record's top-k S3 records and vice versa
4. Embeddings (float16) are saved too, so blocking can score any pair's cosine exactly.

Everything is written straight to S3 (--s3-out); the output is too large for the job
tarball. Validation: fold-0 recall@k of the dense channel is printed per country.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import time

import boto3
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def entity_fold(ids, n_folds=10):
    """Same hash as model.entity_fold, so fold 0 means the matcher's validation fold."""
    return np.fromiter((int(hashlib.blake2b(e.encode(), digest_size=4).hexdigest(), 16) % n_folds
                        for e in ids), dtype=np.int16, count=len(ids))


def text(df):
    return ("query: " + df["business_name"].fillna("") + " | "
            + df["business_address"].fillna("")).tolist()


def s3_put(s3, url, obj_writer):
    if not url.startswith("s3://"):                 # local path: dry runs off SageMaker
        os.makedirs(os.path.dirname(url), exist_ok=True)
        with open(url, "wb") as fh:
            obj_writer(fh)
        return
    bucket, key = url[5:].split("/", 1)
    buf = io.BytesIO()
    obj_writer(buf)
    buf.seek(0)
    s3.upload_fileobj(buf, bucket, key)         # multipart: embedding files reach ~4 GB


def encode(model, tok, texts, bs, max_len, dev, train=False):
    enc = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
    return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1).float()


@torch.no_grad()
def embed_all(model, tok, texts, args, dev):
    model.eval()
    order = np.argsort([len(t) for t in texts])
    out = torch.empty((len(texts), model.config.hidden_size), dtype=torch.float16)
    for s in range(0, len(texts), args.eval_bs):
        idx = order[s:s + args.eval_bs]
        out[torch.from_numpy(idx)] = encode(model, tok, [texts[i] for i in idx],
                                            args.eval_bs, args.max_len, dev).half().cpu()
    return out


@torch.no_grad()
def knn(q, d, k, dev, max_elems=1_500_000_000):
    """Exact top-k inner product of each q row against all d rows, on the GPU.
    Query chunks are sized so the chunk x |d| similarity block stays ~3 GB (fp16)."""
    D = d.to(dev)
    chunk = max(64, max_elems // max(D.shape[0], 1))
    idx, sc = [], []
    for s in range(0, len(q), chunk):
        sims = q[s:s + chunk].to(dev) @ D.T
        v, i = sims.topk(min(k, D.shape[0]), dim=1)
        idx.append(i.cpu()); sc.append(v.float().cpu())
        del sims
    return torch.cat(idx).numpy(), torch.cat(sc).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--eval_bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--link_k", type=int, default=3)
    ap.add_argument("--s3-out", dest="s3_out", required=True)
    args, _ = ap.parse_known_args()
    dev = "cuda"
    torch.manual_seed(0)
    s3 = boto3.client("s3")
    s3_put(s3, f"{args.s3_out}/_started", lambda b: b.write(b"ok"))   # fail fast on permissions
    rec_dir = os.environ.get("SM_CHANNEL_RECORDS", "/opt/ml/input/data/records")
    recs = {(sp, s): pd.read_parquet(f"{rec_dir}/{sp}_source{s}.parquet")
            for sp in ("train", "test") for s in (1, 2, 3)}
    gt = pd.read_parquet(f"{rec_dir}/train_ground_truth.parquet")
    gt_all = gt[entity_fold(gt["source1_entity_id"].values) == 0]      # validation fold, for the recall report

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(dev)

    # ---------------------------------------------------------- fine-tune ---
    gt = gt[gt["matched_entity_ids"] != ""]
    gt = gt[entity_fold(gt["source1_entity_id"].values) != 0]
    s1t = dict(zip(recs["train", 1]["entity_id"], text(recs["train", 1])))
    oth = pd.concat([recs["train", 2], recs["train", 3]])
    otht = dict(zip(oth["entity_id"], text(oth)))
    matches = gt["matched_entity_ids"].str.split(",").tolist()
    anchors = gt["source1_entity_id"].tolist()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(0)
    model.train()
    t0, step = time.time(), 0
    for p in range(args.passes):
        pick = [m[rng.integers(len(m))] for m in matches]          # one match per S1
        order = rng.permutation(len(anchors))
        total = len(order) // args.bs
        for b in range(total):
            idx = order[b * args.bs:(b + 1) * args.bs]
            qa = encode(model, tok, [s1t[anchors[i]] for i in idx], args.bs, args.max_len, dev)
            qb = encode(model, tok, [otht[pick[i]] for i in idx], args.bs, args.max_len, dev)
            logits = qa @ qb.T / args.temp
            y = torch.arange(len(idx), device=dev)
            loss = (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0:
                print(f"pass {p} step {b + 1}/{total} loss {loss.item():.4f} "
                      f"{step * args.bs / (time.time() - t0):,.0f} pairs/s", flush=True)
    del s1t, otht, oth

    # ------------------------------------------------ embed + search + save ---
    for sp in ("train", "test"):
        emb = {}
        for s in (1, 2, 3):
            t = time.time()
            emb[s] = embed_all(model, tok, text(recs[sp, s]), args, dev)
            print(f"embedded {sp} s{s}: {len(emb[s]):,} in {time.time() - t:.0f}s", flush=True)
            s3_put(s3, f"{args.s3_out}/emb/{sp}_s{s}.npy", lambda b: np.save(b, emb[s].numpy()))
        for country in sorted(recs[sp, 1]["country"].unique()):
            m = {s: (recs[sp, s]["country"] == country).to_numpy() for s in (1, 2, 3)}
            ids = {s: recs[sp, s]["entity_id"].to_numpy()[m[s]] for s in (1, 2, 3)}
            e = {s: emb[s][torch.from_numpy(m[s])] for s in (1, 2, 3)}
            if len(ids[1]) == 0:
                continue
            q_ids = np.concatenate([ids[2], ids[3]])
            if len(q_ids):
                i, v = knn(torch.cat([e[2], e[3]]), e[1], args.k, dev)
                pairs = pd.DataFrame({"cand_entity_id": np.repeat(q_ids, i.shape[1]),
                                      "s1_entity_id": ids[1][i.ravel()],
                                      "dense_cos": v.ravel().astype(np.float32)})
                s3_put(s3, f"{args.s3_out}/dense_{sp}_{country}.parquet",
                       lambda b: pairs.to_parquet(b, index=False))
                if sp == "train":
                    s1set = set(ids[1])
                    t_ = gt_all[gt_all["source1_entity_id"].isin(s1set)]
                    true = {(m_, s) for s, ms in zip(t_["source1_entity_id"], t_["matched_entity_ids"].str.split(","))
                            for m_ in ms if m_}
                    fold0 = pairs[pairs["s1_entity_id"].isin(set(t_["source1_entity_id"]))]
                    found = sum((c, s) in true for c, s in zip(fold0["cand_entity_id"], fold0["s1_entity_id"]))
                    n_true = len(true)
                    print(f"[dense recall@{args.k}] train {country} fold 0: {found / max(n_true, 1):.4f} "
                          f"({found:,}/{n_true:,})", flush=True)
            for a, b in ((2, 3), (3, 2)):
                if len(ids[a]) and len(ids[b]):
                    i, v = knn(e[a], e[b], args.link_k, dev)
                    links = pd.DataFrame({"a_id": np.repeat(ids[a], i.shape[1]),
                                          "b_id": ids[b][i.ravel()],
                                          "link_cos": v.ravel().astype(np.float32)})
                    s3_put(s3, f"{args.s3_out}/links_{sp}_{country}_s{a}s{b}.parquet",
                           lambda b_: links.to_parquet(b_, index=False))
            print(f"searched {sp} {country}", flush=True)
        for s in (1, 2, 3):
            s3_put(s3, f"{args.s3_out}/emb/{sp}_s{s}_ids.parquet",
                   lambda b: recs[sp, s][["entity_id"]].to_parquet(b, index=False))
        del emb
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
