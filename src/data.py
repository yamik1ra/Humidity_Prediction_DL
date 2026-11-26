"""Loading, feature engineering, chronological split and windowing."""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    with open(path or ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_frame(cfg, freq):
    """Read csv, sort by time, impute small gaps, build model features.
    Returns (df, feature_cols, cov_cols). cov_cols are calendar features that are
    known in advance, so the decoder may use them for future steps."""
    ds = cfg["datasets"][freq]
    df = pd.read_csv(ROOT / ds["path"], encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    num = cfg["features"]["numeric"]
    df[num + [cfg["features"]["wind_direction"]]] = (
        df[num + [cfg["features"]["wind_direction"]]].ffill().bfill())
    df["precip"] = df["precip"].fillna(0.0)

    wd = np.deg2rad(df[cfg["features"]["wind_direction"]])
    df["wd_sin"], df["wd_cos"] = np.sin(wd), np.cos(wd)

    doy = df["datetime"].dt.dayofyear.values
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    cov = ["doy_sin", "doy_cos"]
    if freq == "hourly":
        hr = df["datetime"].dt.hour.values
        df["hr_sin"], df["hr_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
        cov += ["hr_sin", "hr_cos"]
    feats = num + ["wd_sin", "wd_cos"] + cov
    return df, feats, cov


def split_bounds(n, cfg):
    s = cfg["split"]
    a = int(n * s["train"]); b = a + int(n * s["val"])
    return a, b, n


def make_windows(df, feats, cov, cfg, freq):
    """Build seq2seq samples.
    For a window ending at index t (last observed step):
      enc  X  : features[t-L+1 .. t]                      (L, F)
      dec_cov : calendar features at t+1 .. t+H           (H, C)  known in advance
      y       : humidity[t+1 .. t+H]                      (H,)    target
      y_last  : humidity[t]                               ()      first decoder input
    Samples are assigned to train/val/test by the index of their LAST target step,
    so no target of a validation/test window lies in an earlier split's period
    and no window ever sees the future."""
    target, H, L = cfg["target"], cfg["horizon"], cfg["datasets"][freq]["lookback"]
    n = len(df)
    tr_end, va_end, _ = split_bounds(n, cfg)

    # scaler statistics from the TRAIN rows only (no leakage)
    mu = df.loc[:tr_end - 1, feats].mean(); sd = df.loc[:tr_end - 1, feats].std().replace(0, 1)
    Z = ((df[feats] - mu) / sd).values.astype("float32")
    t_mu, t_sd = float(mu[target]), float(sd[target])
    y_all = Z[:, feats.index(target)]
    C = Z[:, [feats.index(c) for c in cov]]

    ends = np.arange(L - 1, n - H)                       # t
    X = np.stack([Z[t - L + 1:t + 1] for t in ends])
    DC = np.stack([C[t + 1:t + H + 1] for t in ends]).astype("float32")
    Y = np.stack([y_all[t + 1:t + H + 1] for t in ends])
    YL = y_all[ends]
    last_target = ends + H
    part = np.where(last_target < tr_end, 0, np.where(last_target < va_end, 1, 2))
    # drop the few train/val windows whose targets straddle a boundary
    first_target = ends + 1
    straddle = ((part == 1) & (first_target < tr_end)) | ((part == 2) & (first_target < va_end))
    out = {}
    for i, name in enumerate(["train", "val", "test"]):
        m = (part == i) & ~straddle
        out[name] = dict(X=X[m], DC=DC[m], Y=Y[m], YL=YL[m],
                         dates=df["datetime"].values[ends[m] + H])
    out["scale"] = dict(mean=t_mu, std=t_sd)
    out["feats"] = feats
    return out


def inv(y_scaled, scale):
    """Back to humidity %, clipped to the physical range."""
    return np.clip(np.asarray(y_scaled) * scale["std"] + scale["mean"], 0, 100)
