"""Train + evaluate every model on one dataset:  python -m src.train daily|hourly"""
import json, sys, time
import numpy as np
from . import data as D, metrics as M, baselines as B, model as S


def run(freq):
    cfg = D.load_config()
    df, feats, cov = D.load_frame(cfg, freq)
    w = D.make_windows(df, feats, cov, cfg, freq)
    sc, Yt = w["scale"], D.inv(w["test"]["Y"], w["scale"])
    print(freq, {k: len(w[k]["X"]) for k in ("train", "val", "test")}, flush=True)

    preds = {"Persistence": B.persistence(w["test"])}
    preds["Ridge (linear)"] = B.ridge(w)(w["test"])
    out_dir = D.ROOT / "models"
    for name, att in [("Seq2Seq (no attention)", False), ("Seq2Seq + Attention", True)]:
        t0 = time.time()
        m, hist = S.fit(w, cfg, use_attention=att)
        preds[name] = S.predict(m, w["test"])
        m.save_weights(out_dir / f"{freq}_{'attn' if att else 'noattn'}.weights.h5")
        json.dump(hist, open(out_dir / f"{freq}_{'attn' if att else 'noattn'}_history.json", "w"))
        print(f"{name}: {time.time()-t0:.0f}s", flush=True)
        if att:
            np.save(D.ROOT / "reports" / f"{freq}_attention.npy", S.attention_weights(m, w["test"]))

    res = {k: M.evaluate(Yt, D.inv(p, sc)) for k, p in preds.items()}
    np.savez(D.ROOT / "reports" / f"{freq}_test_predictions.npz", y_true=Yt, dates=w["test"]["dates"],
             **{k: D.inv(p, sc) for k, p in preds.items()})
    json.dump(res, open(D.ROOT / "reports" / f"{freq}_results.json", "w"), indent=1)
    for k, r in res.items():
        print(f"{k:26s}", {a: round(b, 3) for a, b in r["overall"].items()}, flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
