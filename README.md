# hcm-humidity-forecast

Forecast Ho Chi Minh City `humidity` for the next 5 consecutive steps with an
Encoder-Decoder (GRU) network + Attention. Metrics: R2, NSE, MAE, RMSE.

**Final deliverable:** `notebooks/final_notebook.ipynb` (end-to-end, used for presenting).

## Result (hourly data, test set, 5 hours pooled, Seq2Seq + Attention)
R2 0.842 | NSE 0.838 | MAE 4.43 | RMSE 6.01 (persistence MAE 8.54, climatology 6.55, Ridge 4.45).
Why hourly rather than daily: see `reports/dataset_selection.md`.

## Layout
- `configs/config.yaml` - all settings
- `data/` - weather_hcm_daily.csv, weather_hcm_hourly.csv
- `src/` - data.py (features, split, windows), metrics.py, baselines.py, model.py, train.py
- `models/` - saved weights + training histories
- `reports/` - test predictions, results json, figures
- `tests/` - metric and no-leakage tests
- `notebooks/final_notebook.ipynb`

## Run
```
pip install -r requirements.txt
python -m pytest tests
python -m src.train hourly      # ~15 min on 1 CPU (daily: python -m src.train daily)
jupyter nbconvert --to notebook --execute --inplace notebooks/final_notebook.ipynb
```
