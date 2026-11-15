"""End-to-end: raw events -> SQL -> features -> train two-tower -> evaluate vs popularity."""
import argparse
import json
import os
import time

import numpy as np
import torch

from . import data, features
from .metrics import evaluate, to_csr
from .model import TwoTower, in_batch_softmax_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "lastfm"], default="synthetic")
    ap.add_argument("--tsv", default="data/synthetic.tsv", help="path to raw events TSV")
    ap.add_argument("--db", default="data/events.db")
    ap.add_argument("--min_track_users", type=int, default=5)
    ap.add_argument("--min_user_tracks", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--n_users", type=int, default=1000, help="synthetic only")
    ap.add_argument("--n_tracks", type=int, default=6000, help="synthetic only")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. data (SQL) -------------------------------------------------------
    if args.source == "synthetic" and not os.path.exists(args.tsv):
        os.makedirs(os.path.dirname(args.tsv) or ".", exist_ok=True)
        data.generate_synthetic(args.tsv, n_users=args.n_users, n_tracks=args.n_tracks, seed=args.seed)
    t = time.time()
    con = data.load_events_to_sqlite(args.tsv, args.db)
    raw, n_events = data.build_interactions(con, args.min_track_users, args.min_user_tracks)
    print(f"[data] {n_events:,} raw events -> {len(raw):,} user-track pairs ({time.time()-t:.0f}s)")

    # ---- 2. features (pandas) ------------------------------------------------
    df = features.encode_and_split(raw)
    n_users, n_items, n_art = df["u"].max() + 1, df["i"].max() + 1, df["a"].max() + 1
    user_f, item_f, item_artist, train = features.build_features(df)
    print(f"[feat] users={n_users:,} tracks={n_items:,} artists={n_art:,} | "
          f"train/val/test pairs = {(df.split=='train').sum():,}/{(df.split=='val').sum():,}/{(df.split=='test').sum():,}")

    sp = {s: df[df.split == s] for s in ["train", "val", "test"]}
    M = {s: to_csr(sp[s]["u"].values, sp[s]["i"].values, n_users, n_items) for s in sp}
    seen_train, seen_trval = M["train"], M["train"] + M["val"]
    # candidate catalogue: tracks that have at least one training interaction
    cand = np.zeros(n_items, dtype=bool); cand[train["i"].unique()] = True
    cand_t = torch.as_tensor(cand, device=dev)

    # ---- 3. model ------------------------------------------------------------
    model = TwoTower(n_users, n_items, n_art, user_f, item_f, out_dim=args.dim).to(dev)
    model.set_item_artist(item_artist)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    pop = np.asarray(M["train"].sum(0)).ravel().astype("float32")        # distinct train listeners
    q = pop / pop.sum()
    log_q = torch.as_tensor(np.log(np.clip(q, 1e-9, None)), device=dev)

    def model_scores(users):
        return (model.encode_users(users) @ item_emb.T).masked_fill(~cand_t, -1e9)

    pop_t = torch.as_tensor(pop, device=dev)
    def pop_scores(users):
        return pop_t[None, :].expand(len(users), -1).masked_fill(~cand_t[None, :], -1e9)

    U = torch.as_tensor(train["u"].values, device=dev)
    I = torch.as_tensor(train["i"].values, device=dev)
    W = torch.as_tensor(train["w"].values, device=dev)

    best, bad, best_state = -1, 0, None
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(U), device=dev); tot = 0.0; t = time.time()
        for s in range(0, len(U) - args.batch + 1, args.batch):
            b = perm[s:s + args.batch]
            loss = in_batch_softmax_loss(model, U[b], I[b], W[b], log_q)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        model.eval()
        item_emb = model.all_item_embeddings()
        v = evaluate(model_scores, n_users, seen_train, M["val"], device=dev)
        print(f"[ep {ep:02d}] loss={tot/max(1,len(U)//args.batch):.3f} val ndcg@10={v['ndcg@10']:.4f} "
              f"recall@20={v['recall@20']:.4f} ({time.time()-t:.0f}s)")
        if v["ndcg@10"] > best:
            best, bad = v["ndcg@10"], 0
            best_state = {k: x.clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience: break

    # ---- 4. final test evaluation -------------------------------------------
    model.load_state_dict(best_state); model.eval()
    item_emb = model.all_item_embeddings()
    res = {
        "dataset": {"raw_events": int(n_events), "users": int(n_users), "tracks": int(n_items),
                    "train_pairs": int((df.split == "train").sum()), "source": args.source},
        "popularity": evaluate(pop_scores, n_users, seen_trval, M["test"], device=dev),
        "two_tower": evaluate(model_scores, n_users, seen_trval, M["test"], device=dev),
    }
    print(json.dumps(res, indent=2))
    json.dump(res, open(args.out, "w"), indent=2)
    torch.save({"state": best_state, "args": vars(args)}, "two_tower.pt")


if __name__ == "__main__":
    main()
