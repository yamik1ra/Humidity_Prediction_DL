"""Reference models: persistence and a linear (Ridge) direct multi-output regressor."""
import numpy as np
from sklearn.linear_model import Ridge


def persistence(d):
    """Repeat the last observed (scaled) humidity for all H steps."""
    return np.repeat(d["YL"][:, None], d["Y"].shape[1], axis=1)


def _flat(d):
    return np.concatenate([d["X"].reshape(len(d["X"]), -1), d["DC"].reshape(len(d["DC"]), -1)], axis=1)


def ridge(data, alpha=10.0):
    m = Ridge(alpha=alpha).fit(_flat(data["train"]), data["train"]["Y"])
    return lambda d: m.predict(_flat(d))


def climatology(df, freq, cfg, w):
    """Seasonal-naive reference, returns predictions in %.
    daily : mean humidity of the same day-of-year (train period only)
    hourly: mean humidity of the same (15-day block, hour-of-day) (train period only)
    Shows how much of the score is just the calendar cycle."""
    import pandas as pd
    from .data import split_bounds
    tr_end, _, _ = split_bounds(len(df), cfg)
    tr, H = df.iloc[:tr_end], cfg["horizon"]
    key = (lambda s: s.dt.dayofyear) if freq == "daily" else (lambda s: s.dt.dayofyear // 15 * 100 + s.dt.hour)
    clim, glob = tr.groupby(key(tr["datetime"]))["humidity"].mean(), tr["humidity"].mean()
    step = pd.Timedelta(days=1) if freq == "daily" else pd.Timedelta(hours=1)
    last = pd.Series(pd.to_datetime(w["test"]["dates"]))          # datetime of the last target step
    cols = [key(last - step * (H - 1 - k)).map(clim).fillna(glob).values for k in range(H)]
    return np.stack(cols, axis=1)
