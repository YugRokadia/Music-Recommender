"""Data layer: raw listening events -> SQLite -> aggregated user/track table.

Real data : Last.fm 1K users dataset (userid-timestamp-artid-artname-traid-traname.tsv)
            http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html
Synthetic : generate_synthetic() writes a file in the *same* format so the whole
            pipeline can be exercised end-to-end without the real download.
"""
import os
import sqlite3

import numpy as np
import pandas as pd

COLS = ["user", "ts", "artist_mbid", "artist", "track_mbid", "track"]


def generate_synthetic(path, n_users=1000, n_tracks=6000, n_artists=600,
                       dim=16, events_per_user=400, seed=0):
    """Latent-factor listening log with a long-tail popularity prior."""
    rng = np.random.default_rng(seed)
    art_vec = rng.normal(size=(n_artists, dim))
    track_artist = rng.integers(0, n_artists, n_tracks)
    track_vec = art_vec[track_artist] + 0.5 * rng.normal(size=(n_tracks, dim))
    log_pop = np.log(1.0 / (np.arange(1, n_tracks + 1) ** 0.8))
    rng.shuffle(log_pop)
    user_vec = rng.normal(size=(n_users, dim))
    t0, t1 = pd.Timestamp("2008-01-01").value // 10**9, pd.Timestamp("2009-06-01").value // 10**9

    rows = []
    for u in range(n_users):
        logits = 0.6 * (track_vec @ user_vec[u]) / np.sqrt(dim) * 3 + 0.7 * log_pop
        p = np.exp(logits - logits.max()); p /= p.sum()
        n = max(50, int(rng.normal(events_per_user, 100)))
        items = rng.choice(n_tracks, size=n, p=p)
        # users "discover" tracks over time: first-listen order is random
        ts = np.sort(rng.integers(t0, t1, size=n))
        for i, t in zip(items, ts):
            rows.append((f"user_{u:06d}", pd.Timestamp(int(t), unit="s").strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "", f"artist_{track_artist[i]}", "", f"track_{i}"))
    pd.DataFrame(rows, columns=COLS).to_csv(path, sep="\t", header=False, index=False)
    return path


def load_events_to_sqlite(tsv_path, db_path, chunksize=1_000_000):
    """Stream the raw TSV into SQLite (works for the ~19M-row Last.fm 1K file)."""
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    first = True
    n_loaded = 0
    for chunk in pd.read_csv(tsv_path, sep="\t", header=None, names=COLS, chunksize=chunksize,
                             quoting=3, on_bad_lines="skip", dtype=str,
                             encoding="utf-8", encoding_errors="replace"):
        chunk = chunk.dropna(subset=["user", "ts", "artist", "track"])
        chunk.to_sql("events", con, if_exists="replace" if first else "append", index=False)
        first = False
        n_loaded += len(chunk)
        print(f"[data] loaded {n_loaded:,} events", flush=True)
    con.execute("CREATE INDEX idx_events_user ON events(user)")
    con.commit()
    return con


AGG_SQL = """
DROP TABLE IF EXISTS user_track;
CREATE TABLE user_track AS
SELECT user, artist, track,
       COUNT(*)  AS plays,
       MIN(ts)   AS first_ts,
       MAX(ts)   AS last_ts
FROM events
GROUP BY user, artist, track;

DROP TABLE IF EXISTS track_stats;
CREATE TABLE track_stats AS
SELECT artist, track,
       COUNT(DISTINCT user) AS n_users,
       SUM(plays)           AS total_plays
FROM user_track
GROUP BY artist, track
HAVING COUNT(DISTINCT user) >= :min_track_users;

DROP TABLE IF EXISTS user_stats;
CREATE TABLE user_stats AS
SELECT ut.user, COUNT(*) AS n_tracks
FROM user_track ut JOIN track_stats ts USING (artist, track)
GROUP BY ut.user
HAVING COUNT(*) >= :min_user_tracks;
"""

FETCH_SQL = """
SELECT ut.user, ut.artist, ut.track, ut.plays, ut.first_ts, ut.last_ts
FROM user_track ut
JOIN track_stats ts USING (artist, track)
JOIN user_stats  us USING (user)
"""

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _to_epoch_seconds(s):
    """ISO-8601 strings -> epoch seconds (float). Unparseable values become NaN."""
    dt = pd.to_datetime(s, utc=True, format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
    return (dt - _EPOCH).dt.total_seconds()


def build_interactions(con, min_track_users=5, min_user_tracks=20):
    """SQL-side aggregation + filtering. Returns (interactions df, n_raw_events)."""
    n_events = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    print("[data] aggregating in SQL (this can take several minutes)...", flush=True)
    con.executescript(AGG_SQL.replace(":min_track_users", str(int(min_track_users)))
                             .replace(":min_user_tracks", str(int(min_user_tracks))))
    df = pd.read_sql_query(FETCH_SQL, con)
    df["first_ts"] = _to_epoch_seconds(df["first_ts"])
    df["last_ts"] = _to_epoch_seconds(df["last_ts"])
    df = df.dropna(subset=["first_ts", "last_ts"]).reset_index(drop=True)
    df["first_ts"] = df["first_ts"].astype("int64")
    df["last_ts"] = df["last_ts"].astype("int64")
    return df, n_events