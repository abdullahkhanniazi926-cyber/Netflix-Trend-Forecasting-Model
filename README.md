\# Netflix Trend Forecasting Model



A time series forecasting model that predicts future Netflix content release trends based on historical monthly addition patterns — built as Task 5 of the Auspify Machine Learning internship.



\## What It Does



Analyzes 14 years of Netflix content addition history (2008-2021) and forecasts how many titles will likely be added in each of the next 6 months.



\## How It Works



1\. \*\*Time-Based Feature Preparation\*\* — Converted `date\_added` to real datetime objects, extracted year/month, and dropped unparseable dates (none were dropped — all 8,790 rows had valid dates).

2\. \*\*Historical Pattern Analysis\*\* — Aggregated titles into a monthly time series and plotted it. Found near-zero activity from 2008-2014, explosive growth from 2015-2019 (82 to 2,016 titles/year), then a slight dip in 2020-2021 (partly an artifact of 2021 being a partial year in the dataset).

3\. \*\*Model Building\*\* — Trained a Holt-Winters Exponential Smoothing model (additive trend, additive seasonality, 12-month period) on data through August 2021 (trimmed to exclude the partial final month), holding back the last 6 months as a test set.

4\. \*\*Future Predictions\*\* — Refit the model on the full available time series and forecasted 6 months beyond the dataset's end date (Sept 2021 - Feb 2022).

5\. \*\*Accuracy Evaluation\*\* — Measured forecast accuracy against the held-out test period using MAE, RMSE, and MAPE.



\## Tech Stack



\- Python

\- pandas, numpy

\- statsmodels (`ExponentialSmoothing`)

\- scikit-learn (`mean\_absolute\_error`, `mean\_squared\_error`)

\- matplotlib



\## Dataset



`Dataset.csv` — 8,790 Netflix titles (same dataset as Tasks 1-4), aggregated by `date\_added` into a monthly time series of 107 months.



\## How to Run



pip install pandas numpy statsmodels scikit-learn matplotlib

python Forecasting.py





Note: the script opens two matplotlib plot windows (historical trend, then historical + forecast) — each pauses script execution until closed.



\## Results



| Metric | Value |

|---|---|

| MAE | 64.82 titles/month |

| RMSE | 76.53 titles/month |

| MAPE | 33.14% |



The model found \*\*no meaningful seasonality\*\* in the data (seasonal smoothing parameter optimized to 0) — Netflix's monthly additions are driven almost entirely by the overall growth trend, not a repeating yearly pattern.



\## Forecast (6 months beyond dataset)



| Month | Predicted Titles |

|---|---|

| Sep 2021 | 195.0 |

| Oct 2021 | 215.2 |

| Nov 2021 | 210.2 |

| Dec 2021 | 222.4 |

| Jan 2022 | 214.0 |

| Feb 2022 | 194.4 |



\## Known Limitations



\- 33.14% MAPE indicates the model is directionally useful (correctly captures the upward trend) but not reliable for precise monthly planning, since actual month-to-month releases are highly volatile (e.g. swinging from 112 to 257 titles within 4 months).

\- Exponential smoothing trades responsiveness for stability — it underestimates sharp spikes rather than chasing them, which is reflected in RMSE being notably higher than MAE (a few large misses, not just consistent small ones).

\- 2021 is a partial year in the source data (through September only), which can visually resemble a slowdown that may not reflect the true trend.

