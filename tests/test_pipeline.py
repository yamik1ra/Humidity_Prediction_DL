import numpy as np
from sklearn.metrics import r2_score
from src import metrics, data


def test_nse_equals_sklearn_r2():
    rng = np.random.default_rng(0); y = rng.normal(size=500); p = y + rng.normal(scale=.5, size=500)
    assert abs(metrics.nse(y, p) - r2_score(y, p)) < 1e-9


def test_perfect_forecast():
    y = np.random.default_rng(1).normal(size=(50, 5))
    r = metrics.evaluate(y, y)["overall"]
    assert r["MAE"] == 0 and r["RMSE"] == 0 and abs(r["NSE"] - 1) < 1e-12 and abs(r["R2"] - 1) < 1e-12


def test_no_leakage_between_splits():
    cfg = data.load_config()
    df, feats, cov = data.load_frame(cfg, "daily")
    w = data.make_windows(df, feats, cov, cfg, "daily")
    assert w["train"]["dates"].max() < w["val"]["dates"].min() < w["val"]["dates"].max() < w["test"]["dates"].min()
    assert w["train"]["X"].shape[1:] == (cfg["datasets"]["daily"]["lookback"], len(feats))
    assert w["train"]["Y"].shape[1] == cfg["horizon"]


def test_target_alignment():
    cfg = data.load_config()
    df, feats, cov = data.load_frame(cfg, "daily")
    w = data.make_windows(df, feats, cov, cfg, "daily")["train"]
    # y_last must equal the humidity of the last encoder row
    assert np.allclose(w["YL"], w["X"][:, -1, feats.index("humidity")])
