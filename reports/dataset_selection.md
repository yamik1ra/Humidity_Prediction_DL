# Why the hourly dataset was chosen (supporting evidence, not part of the presentation notebook)

Both files were run through the identical pipeline (same features, chronological 70/15/15 split,
same model code). Test-set results, 5 steps pooled:

| Model | Hourly R2 | Hourly NSE | Hourly MAE | Hourly RMSE | Daily R2 | Daily NSE | Daily MAE | Daily RMSE |
|---|---|---|---|---|---|---|---|---|
| Persistence | 0.475 | 0.378 | 8.540 | 11.785 | 0.423 | 0.307 | 5.532 | 7.243 |
| Climatology | 0.674 | 0.670 | 6.549 | 8.578 | 0.517 | 0.508 | 4.834 | 6.100 |
| Ridge (linear) | 0.836 | 0.834 | 4.451 | 6.080 | 0.523 | 0.494 | 4.888 | 6.184 |
| Seq2Seq (no attention) | 0.841 | 0.830 | 4.578 | 6.154 | 0.567 | 0.558 | 4.543 | 5.782 |
| **Seq2Seq + Attention** | **0.842** | **0.838** | **4.432** | 6.008 | 0.590 | 0.576 | 4.497 | **5.665** |

Best model per dataset: hourly wins R2, NSE and MAE; daily wins RMSE (3 of 4 metrics for hourly).

Caveats: "5 steps" is 5 hours vs 5 days; R2/NSE depend on target variance (hourly std 16.1 %, daily 9.7 %),
so relative gains were also checked: against climatology the hourly model cuts MAE by 32 % (daily 7 %),
against persistence by 48 % (daily 19 %). Even Ridge on hourly data (NSE 0.834) beats the best daily model.
MAE is practically tied (4.43 vs 4.50). Single training seed.

Reproduce the daily numbers with `python -m src.train daily` (results in `reports/daily_*`).
