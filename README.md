# Music Recommender: Two-Tower Model

PyTorch two-tower retrieval model that learns user and track embeddings from listening
logs, with SQL aggregation, pandas feature engineering, and a full-catalogue evaluation
against a popularity baseline.

```
raw events ──► SQLite (SQL: aggregate, filter) ──► pandas features + temporal split
           ──► two-tower (in-batch softmax + logQ correction) ──► NDCG@10 / Recall@20 vs popularity
```

## Run on real data (what the résumé numbers should come from)
```bash
pip install -r requirements.txt
bash scripts/download_lastfm.sh
python run.py --source lastfm \
  --tsv data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv \
  --epochs 20 --batch 4096
cat results.json
```
Last.fm 1K has ~19M events from 992 users. Plug the printed numbers into the résumé bullets:
`raw_events` → [N] million, and `popularity` vs `two_tower` → [X] → [Y].

## Smoke test (no download)
`python run.py` generates a synthetic latent-factor log in the same TSV format and runs the whole pipeline.
**Do not quote synthetic-data numbers on a résumé**; they only prove the code works.

## Design
- **SQL (`src/data.py`)**: streams events into SQLite, `GROUP BY user, artist, track` for play count /
  first / last listen, filters tracks with < 5 listeners and users with < 20 tracks.
- **Features (`src/features.py`)**: user (history size, total plays, mean log plays, recency, history span,
  artist entropy) and track (listener count, total plays, mean log plays, recency, track age). Statistics use
  **train rows only**. Training pairs are weighted by log play count × recency decay.
- **Model (`src/model.py`)**: user tower = ID embedding + dense features → MLP; track tower = ID + artist
  embedding + dense features → MLP; L2-normalised, learnable temperature. Loss: in-batch sampled softmax
  with logQ popularity correction.
- **Evaluation (`src/metrics.py`)**: per-user temporal split by first-listen time (70/10/20 train/val/test),
  so the test set is *tracks the user discovers later*. Full ranking over the catalogue, excluding already-heard
  tracks. Binary-relevance NDCG@10 and Recall@20 (hits / #relevant). Popularity baseline ranks by number of train
  listeners. Early stopping on validation NDCG@10; test touched once.

## Caveats / extensions
- Candidate catalogue = tracks with ≥ 1 training listen (applies to both models).
- Last.fm 1K has no genre tags; "track metadata" here is artist + derived stats. Add tags via the Last.fm API
  or use LFM-2b for richer metadata.
- Next steps: ANN retrieval (FAISS) for serving, sequence-aware user tower, hyperparameter sweep.
