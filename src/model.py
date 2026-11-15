import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    """ID embedding (+ optional side-id embedding) + dense features -> MLP -> unit vector."""

    def __init__(self, n_ids, dense_dim, emb_dim=64, out_dim=64, hidden=128,
                 n_side=0, side_dim=16, dropout=0.1):
        super().__init__()
        self.id_emb = nn.Embedding(n_ids, emb_dim)
        self.side_emb = nn.Embedding(n_side, side_dim) if n_side else None
        in_dim = emb_dim + dense_dim + (side_dim if n_side else 0)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, out_dim))
        nn.init.normal_(self.id_emb.weight, std=0.1)

    def forward(self, ids, dense, side=None):
        parts = [self.id_emb(ids), dense]
        if self.side_emb is not None:
            parts.append(self.side_emb(side))
        return F.normalize(self.mlp(torch.cat(parts, -1)), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, n_artists, user_dense, item_dense, out_dim=64, init_temp=0.07):
        super().__init__()
        self.user_tower = Tower(n_users, user_dense.shape[1], out_dim=out_dim)
        self.item_tower = Tower(n_items, item_dense.shape[1], out_dim=out_dim, n_side=n_artists)
        self.register_buffer("user_dense", torch.as_tensor(user_dense))
        self.register_buffer("item_dense", torch.as_tensor(item_dense))
        self.register_buffer("item_artist", torch.zeros(n_items, dtype=torch.long))
        self.log_temp = nn.Parameter(torch.tensor(float(init_temp)).log())

    def set_item_artist(self, item_artist):
        self.item_artist.copy_(torch.as_tensor(np.array(item_artist), dtype=torch.long))

    def encode_users(self, u):
        return self.user_tower(u, self.user_dense[u])

    def encode_items(self, i):
        return self.item_tower(i, self.item_dense[i], self.item_artist[i])

    @torch.no_grad()
    def all_item_embeddings(self, bs=65536):
        n = self.item_dense.shape[0]
        return torch.cat([self.encode_items(torch.arange(s, min(s + bs, n), device=self.item_dense.device))
                          for s in range(0, n, bs)])

    @torch.no_grad()
    def all_user_embeddings(self, bs=65536):
        n = self.user_dense.shape[0]
        return torch.cat([self.encode_users(torch.arange(s, min(s + bs, n), device=self.user_dense.device))
                          for s in range(0, n, bs)])


def in_batch_softmax_loss(model, u, i, w, log_q):
    """Sampled-softmax with in-batch negatives and logQ popularity correction
    (Yi et al., 2019), confidence-weighted. Same-item collisions are masked."""
    zu, zi = model.encode_users(u), model.encode_items(i)
    temp = model.log_temp.exp().clamp(0.01, 1.0)
    logits = zu @ zi.T / temp - log_q[i][None, :]
    same = i[:, None] == i[None, :]
    logits = logits.masked_fill(same & ~torch.eye(len(i), dtype=torch.bool, device=i.device), -1e9)
    loss = F.cross_entropy(logits, torch.arange(len(i), device=i.device), reduction="none")
    return (loss * w).sum() / w.sum()
