"""pandas feature engineering + temporal per-user split.

All user/track statistics are computed from TRAIN interactions only (no leakage).
"""
import numpy as np
import pandas as pd

DAY = 86400.0


def encode_and_split(df, val_frac=0.1, test_frac=0.2, min_split_tracks=10):
    """Dense ids + per-user temporal split by first-listen time.

    For every user, tracks are ordered by first listen: the oldest ~70% are train,
    the next ~10% validation, the newest ~20% test. Test therefore measures
    *discovery* of tracks the user had not heard before.
    """
    df = df.copy()
    df["u"] = pd.factorize(df["user"])[0]
    df["track_key"] = df["artist"] + " || " + df["track"]
    df["i"] = pd.factorize(df["track_key"])[0]
    df["a"] = pd.factorize(df["artist"])[0]
    df = df.sort_values(["u", "first_ts", "i"]).reset_index(drop=True)
    rank = df.groupby("u").cumcount()
    n = df.groupby("u")["i"].transform("size")
    pos = (rank + 0.5) / n
    df["split"] = np.where(pos < 1 - val_frac - test_frac, "train",
                    np.where(pos < 1 - test_frac, "val", "test"))
    df.loc[n < min_split_tracks, "split"] = "train"
    return df


def build_features(df):
    """Returns user_feats, item_feats (float32 arrays), item_artist (int array), plus weights."""
    train = df[df["split"] == "train"].copy()
    n_users, n_items = df["u"].max() + 1, df["i"].max() + 1
    t_end = float(train["last_ts"].max())
    train["log_plays"] = np.log1p(train["plays"])
    train["age_d"] = (t_end - train["last_ts"]) / DAY            # recency of last play
    train["since_first_d"] = (t_end - train["first_ts"]) / DAY

    # ---- user features -------------------------------------------------------
    g = train.groupby("u")
    uf = pd.DataFrame({
        "log_n_tracks": np.log1p(g.size()),
        "log_total_plays": np.log1p(g["plays"].sum()),
        "mean_log_plays": g["log_plays"].mean(),
        "mean_recency_d": np.log1p(g["age_d"].mean()),
        "history_span_d": np.log1p(g["since_first_d"].max()),
    })
    # artist diversity: normalised entropy of play share across artists
    ap = train.groupby(["u", "a"])["plays"].sum().rename("p").reset_index()
    ap["share"] = ap["p"] / ap.groupby("u")["p"].transform("sum")
    ap["h"] = -ap["share"] * np.log(ap["share"])
    uf["artist_entropy"] = ap.groupby("u")["h"].sum()
    uf = uf.reindex(range(n_users)).fillna(0.0)

    # ---- track features ------------------------------------------------------
    g = train.groupby("i")
    itf = pd.DataFrame({
        "log_n_users": np.log1p(g["u"].nunique()),
        "log_total_plays": np.log1p(g["plays"].sum()),
        "mean_log_plays": g["log_plays"].mean(),
        "mean_recency_d": np.log1p(g["age_d"].mean()),
        "track_age_d": np.log1p(g["since_first_d"].max()),
    })
    itf = itf.reindex(range(n_items)).fillna(0.0)

    def zscore(x):
        x = x.values.astype("float32")
        return (x - x.mean(0)) / (x.std(0) + 1e-6)

    item_artist = df.drop_duplicates("i").sort_values("i")["a"].values
    # confidence weight: heavier plays and more recent listens count more
    w = train["log_plays"] + 1.0
    w = w * (0.5 + 0.5 * np.exp(-train["age_d"] / 180.0))
    train["w"] = (w / w.mean()).astype("float32")
    return zscore(uf), zscore(itf), item_artist, train
