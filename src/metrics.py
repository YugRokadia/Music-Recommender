import numpy as np
import torch
import scipy.sparse as sp


def to_csr(u, i, n_users, n_items):
    return sp.csr_matrix((np.ones(len(u), dtype=np.float32), (u, i)), shape=(n_users, n_items))


def evaluate(score_fn, n_users, mask_csr, target_csr, ks=(10, 20), user_batch=512, device="cpu"):
    """Full-catalogue ranking. score_fn(user_ids) -> [B, n_items] tensor.
    `mask_csr` items (already seen) are excluded; `target_csr` items are the positives.
    Reports NDCG@10 and Recall@20 (binary relevance, Recall = hits / #relevant)."""
    users = np.where(target_csr.getnnz(axis=1) > 0)[0]
    kmax = max(ks)
    disc = 1.0 / np.log2(np.arange(2, kmax + 2))
    ndcg = {k: [] for k in ks}
    rec = {k: [] for k in ks}
    for s in range(0, len(users), user_batch):
        ub = users[s:s + user_batch]
        scores = score_fn(torch.as_tensor(ub, device=device)).float()
        m = torch.as_tensor(mask_csr[ub].toarray(), device=device).bool()
        scores = scores.masked_fill(m, -1e9)
        top = scores.topk(kmax, dim=1).indices.cpu().numpy()
        tgt = target_csr[ub].toarray() > 0
        hit = np.take_along_axis(tgt, top, axis=1)
        n_rel = tgt.sum(1)
        for k in ks:
            dcg = (hit[:, :k] * disc[:k]).sum(1)
            idcg = np.array([disc[:min(r, k)].sum() for r in n_rel])
            ndcg[k].append(dcg / idcg)
            rec[k].append(hit[:, :k].sum(1) / n_rel)
    out = {}
    for k in ks:
        out[f"ndcg@{k}"] = float(np.concatenate(ndcg[k]).mean())
        out[f"recall@{k}"] = float(np.concatenate(rec[k]).mean())
    out["n_eval_users"] = int(len(users))
    return out
