# Netflix (NFLX) Stock Price Prediction — v1

> **A time-series forecasting study on Netflix stock prices (2002–2025) using classical statistical models and deep learning. This version proves the hypothesis that raw stock prices are fundamentally unpredictable, empirically validating the Random Walk Theory and the Efficient Market Hypothesis (EMH). Version 2 of this project addresses this limitation by predicting absolute returns instead of raw prices.**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Stationarity Analysis](#stationarity-analysis)
4. [Baseline Model — Naïve Forecast](#baseline-model--naïve-forecast)
5. [ARIMA Model](#arima-model)
6. [Deep Learning Models](#deep-learning-models)
   - [Univariate Dense Networks (Window-based)](#univariate-dense-networks-window-based)
   - [Univariate LSTM Networks](#univariate-lstm-networks)
   - [Multivariate Model with Technical Indicators](#multivariate-model-with-technical-indicators)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results Summary](#results-summary)
9. [Key Finding: Random Walk Theory & EMH](#key-finding-random-walk-theory--emh)
10. [Why This Version Fails — and What v2 Fixes](#why-this-version-fails--and-what-v2-fixes)
11. [Tech Stack](#tech-stack)
12. [Project Structure](#project-structure)

---

## Project Overview

This project attempts to forecast Netflix's daily closing stock price using historical price data from May 2002 to December 2025 (5,940 trading days). A range of approaches is implemented — from a simple naïve forecast to multi-layer stacked LSTMs with technical indicators — to rigorously test whether any ML model can beat a trivial baseline on raw stock price data.

The central conclusion of v1 is **negative**: no model significantly outperforms the naïve forecast. This is not a failure of implementation — it is a meaningful empirical result that directly supports the **Random Walk Theory** and the **Efficient Market Hypothesis (EMH)**. This project serves as a formal proof-of-concept for those theories before attempting a more statistically sound approach in v2.

---

## Dataset

- **Source:** Yahoo Finance via the `yfinance` Python library
- **Ticker:** `NFLX`
- **Date Range:** 2002-05-23 to 2025-12-30
- **Total Observations:** 5,940 trading days
- **Target Variable:** Daily adjusted closing price (`Close`)
- **Train/Test Split:** 80% train (4,752 observations) / 20% test (1,188 observations)

**Why this split ratio?** An 80/20 split is the standard for time-series data where the model must be evaluated on unseen *future* data. Crucially, no shuffling is performed — temporal order is strictly preserved to prevent look-ahead bias, which would artificially inflate model performance.

---

## Stationarity Analysis

Before modelling, the statistical properties of the series are examined using the **Augmented Dickey-Fuller (ADF) test**.

### Raw Price Series (Non-Stationary)

```
ADF Statistic:  2.615
p-value:        0.999
```

The p-value of ~1.0 is far above any significance threshold (0.05), meaning we **fail to reject the null hypothesis of a unit root**. The raw price series is non-stationary — its mean and variance change over time, making it unsuitable for direct modelling with most statistical and ML methods.

### First-Differenced Series (Stationary)

```
ADF Statistic:  -12.086
p-value:        2.18e-22
```

After first-order differencing, the p-value collapses to essentially zero. The series is now stationary — confirming that **price changes (not price levels)** are the appropriate quantity to model. This is a core motivation for switching to absolute returns in v2.

**Why does non-stationarity matter?**
A non-stationary series violates the fundamental assumptions of ARIMA, standard regression, and neural network training alike. Models trained on non-stationary data learn the local trend rather than any true predictive pattern, resulting in a deceptively low training loss but poor generalisation.

---

## Baseline Model — Naïve Forecast

The naïve forecast predicts that tomorrow's price equals today's price:

```
ŷ(t) = y(t - 1)
```

This is the canonical baseline for financial time series and is surprisingly hard to beat. Its MASE (Mean Absolute Scaled Error) score of **~1.0** defines the unit of comparison for all other models — any model with MASE > 1 is *worse* than doing nothing.

| Metric | Value |
|--------|-------|
| MAE    | 0.939 |
| RMSE   | 1.458 |
| MAPE   | 1.764% |
| MASE   | **0.999** |

---

## ARIMA Model

**Configuration:** `ARIMA(1, 1, 1)` — chosen by inspecting ACF and PACF plots of the first-differenced series, and confirmed by comparing AIC scores.

| Model | AIC |
|-------|-----|
| ARIMA(1,1,1) | 5887.74 |
| ARIMA(1,1,0) | 5906.19 |

The lower AIC of ARIMA(1,1,1) confirms it as the better-fitting specification. A **walk-forward validation** strategy is used: the model is re-fitted at every step with the expanding history window rather than making multi-step forecasts from a fixed origin. This avoids compounding forecast errors and represents best practice for ARIMA in production.

| Metric | Value |
|--------|-------|
| MAE    | 0.945 |
| RMSE   | 1.462 |
| MAPE   | 1.777% |
| MASE   | **1.007** |

ARIMA(1,1,1) barely edges out the naïve baseline — a MASE of 1.007 means it is marginally *worse*. This is consistent with the Random Walk Hypothesis: the first-differenced stock price resembles white noise, with no exploitable autocorrelation structure.

---

## Deep Learning Models

### Univariate Dense Networks (Window-based)

A sliding-window approach converts the 1D price series into supervised learning inputs. Three window sizes are evaluated to find the optimal lookback horizon:

| Model | Window Size | MAE | RMSE | MAPE | MASE |
|-------|-------------|-----|------|------|------|
| Model 1 (Dense) | 7 days  | 13.75 | 21.14 | 1.79% | **1.030** |
| Model 2 (Dense) | 10 days | 14.13 | 21.54 | 1.87% | **1.060** |
| Model 3 (Dense) | 14 days | 14.29 | 21.66 | 1.88% | **1.070** |

**Architecture:**
```
Dense(128, activation='relu') → Dense(1)
```

**Key observations:**
- The 7-day window performs best among the three, suggesting any historical information beyond one week adds noise rather than signal for raw prices.
- All three models have MASE > 1, meaning they are **all worse than the naïve forecast** in absolute terms.
- The large discrepancy between training MAE (< 0.5) and test MAE (13+) is a hallmark of a model that has learned to memorise the training distribution's upward trend rather than any generalisable pattern — a direct consequence of training on a non-stationary series.

**Why `StandardScaler` is fitted only on training data:** The scaler is fit exclusively on `y1` (the training set) and then applied to `y2` (the test set). This prevents data leakage — using future statistics to normalise past data — which would be a form of look-ahead bias.

---

### Univariate LSTM Networks

Two stacked LSTM architectures are tested on the 7-day window (best from the Dense experiment):

**Model `uni_lstm1` — ReLU activation:**
```
Lambda(expand_dims) → LSTM(32, relu, recurrent_dropout=0.2, return_sequences=True)
                    → LSTM(32, relu, recurrent_dropout=0.2) → Dense(1)
```

**Model `uni_lstm2` — Tanh activation:**
```
Lambda(expand_dims) → LSTM(32, tanh, recurrent_dropout=0.2, return_sequences=True)
                    → LSTM(32, tanh, recurrent_dropout=0.2) → Dense(1)
```

| Model | Activation | MAE | RMSE | MAPE | MASE |
|-------|-----------|-----|------|------|------|
| LSTM 1 | ReLU | 40.05 | 51.17 | 5.08% | **3.001** |
| LSTM 2 | Tanh | 212.32 | 376.30 | 15.84% | **15.91** |

**Why LSTMs perform so poorly here:**

- **ReLU in LSTMs is pathological**: LSTMs were designed with `tanh`/`sigmoid` gate activations. Substituting `relu` for the recurrent activation disrupts the gating mechanism and can lead to exploding internal states — reflected here by a MASE of 3.0, three times worse than naïve.
- **Tanh LSTM diverges completely**: The tanh LSTM produces a MASE of ~15.9, indicating near-total divergence. This is again a symptom of training a recurrent network on a non-stationary, noisy, trending signal with no real autocorrelation. The LSTM has nothing to learn from the price sequence and fails catastrophically at generalisation.
- `recurrent_dropout=0.2` is used as a regularisation technique specific to RNNs. Unlike standard dropout (which zeros random inputs), recurrent dropout applies the same mask at each time step, avoiding disruption of the temporal flow of gradients.

---

### Multivariate Model with Technical Indicators

This is the most sophisticated setup in v1, incorporating two technical indicators alongside raw price:

**Feature Engineering:**
- **EMA-20 (20-day Exponential Moving Average):** A trend-following indicator that weights recent prices more heavily, smoothing short-term noise.
- **MACD Histogram (12/26/9 configuration):** Measures momentum by comparing short and long-term EMAs. The histogram captures the acceleration of price momentum.

Each of these three features is lagged by 7 periods, creating a `[7 × 3]` input matrix (21 features total per sample). Two separate `StandardScaler` instances are used — one for inputs and one for the target — to avoid scale contamination between features and labels.

**Multivariate LSTM (`model_3_lstm`):**
```
Lambda(reshape → [batch, 7, 3]) → LSTM(32, relu, return_sequences=True)
                                 → LSTM(32, relu) → Dense(1)
```

**Multivariate Dense (`model_4`):**
```
Dense(128, relu) → Dense(1)
```

| Model | MAE | RMSE | MAPE | MASE |
|-------|-----|------|------|------|
| Multivariate LSTM | 2.32 | 3.47 | 4.45% | **2.470** |
| Multivariate Dense | 1.11 | 1.64 | 2.17% | **1.177** |

The multivariate Dense network (`model_4`) shows the best MAE of all deep learning models (1.11), yet its MASE of 1.177 still fails to beat the naïve baseline. The addition of EMA and MACD provides some marginal improvement, but the models are ultimately fitting to highly autocorrelated technical indicators derived from the same non-stationary price series — not to any genuinely predictive signal.

---

## Evaluation Metrics

Four complementary metrics are used to evaluate all models:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | `mean(|y_true - y_pred|)` | Average absolute error in raw price units (USD). Scale-dependent. |
| **RMSE** | `sqrt(mean((y_true - y_pred)²))` | Penalises large errors more heavily than MAE. Sensitive to outliers. |
| **MAPE** | `mean(|y_true - y_pred| / y_true) × 100` | Percentage error — scale-invariant but undefined at zero and biased for small values. |
| **MASE** | `MAE / MAE_naïve` | **The primary metric.** A MASE < 1 beats the naïve baseline; MASE = 1 equals it; MASE > 1 is worse. Scale-invariant and directly interpretable. |

**Why MASE is the primary metric:** In financial forecasting, the question is not "how large is the error in dollars?" but "does this model add any value over the simplest possible strategy?" MASE directly answers this question by normalising error against the naïve forecast.

---

## Results Summary

| Model | Window / Config | MAE | MASE | Beats Naïve? |
|-------|----------------|-----|------|:---:|
| Naïve Forecast | — | 0.939 | 1.000 | — (baseline) |
| ARIMA(1,1,1) | Walk-forward | 0.945 | 1.007 | ❌ |
| Dense (Univariate) | 7-day | 13.75 | 1.030 | ❌ |
| Dense (Univariate) | 10-day | 14.13 | 1.060 | ❌ |
| Dense (Univariate) | 14-day | 14.29 | 1.070 | ❌ |
| LSTM (ReLU) | 7-day | 40.05 | 3.001 | ❌ |
| LSTM (Tanh) | 7-day | 212.32 | 15.91 | ❌ |
| Multivariate LSTM | 7-day × 3 features | 2.32 | 2.470 | ❌ |
| Multivariate Dense | 7-day × 3 features | 1.11 | **1.177** | ❌ |

**No model in v1 beats the naïve forecast on MASE.**

---

## Key Finding: Random Walk Theory & EMH

The failure of every model in this project is not a bug — it is the expected result according to two foundational theories in financial economics.

### Random Walk Theory

The Random Walk Hypothesis (Malkiel, 1973) states that stock price changes are serially independent — each price movement is statistically unrelated to past movements. Formally:

```
P(t) = P(t-1) + ε(t),    where ε(t) ~ white noise
```

If this holds, then the best prediction of tomorrow's price is simply today's price — exactly what the naïve model does. The ADF test results in this project directly corroborate this: after first differencing, the series resembles white noise with no significant autocorrelation at any lag (as confirmed by the ACF/PACF plots). There is no exploitable pattern.

### Efficient Market Hypothesis (EMH)

The EMH (Fama, 1970) asserts that asset prices at any time fully reflect all available information. In its **weak form**, it specifically claims that no trading strategy based solely on historical price data can consistently generate excess returns — because all such information is already priced in.

Every model in this project uses only historical price and price-derived features (EMA, MACD). The fact that none outperforms the naïve forecast is direct empirical evidence in support of the weak-form EMH for NFLX over this period.

**In short:** This project does not fail to predict stock prices. It *succeeds* in demonstrating that raw stock prices cannot be predicted from their own history, which is precisely what theory predicts.

---

## Why This Version Fails — and What v2 Fixes

### The Root Cause: Non-Stationarity

The core technical problem is that raw closing prices are **non-stationary**. They have a time-varying mean (the long-term upward trend of NFLX from $0.12 to $94) and a time-varying variance (volatility clusters). Models trained on non-stationary data do not learn generalisable patterns — they learn the local trend of the training set and extrapolate it, which fails on the test set.

This is why models show low training loss but drastically higher test loss: the distributional shift between the pre-2021 training data and the post-2021 test data (which includes NFLX's peak at ~$700 and its 2022 crash) is enormous.

### The v2 Fix: Absolute Returns

In **Version 2**, raw prices are replaced with **absolute returns** (day-over-day price differences) as the prediction target:

```
returns(t) = P(t) - P(t-1)
```

This is exactly the first-differencing operation whose stationarity is already confirmed in this notebook (ADF p ≈ 2e-22). By shifting the target from price levels to price changes, the input distribution becomes stable across the entire training period — the model no longer has to generalise across a series that ranged from $0.12 to $694.

| Property | Raw Price | Absolute Return |
|----------|-----------|-----------------|
| Stationarity | ❌ Non-stationary | ✅ Stationary |
| Scale | Varies ($0.12 → $694) | Stable (centred near 0) |
| Economic meaning | Absolute price level | Dollar-denominated daily change |
| Suitable for ML | ❌ No | ✅ Yes |
| Additive over time | ❌ No | ✅ Yes |

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `yfinance` | latest | Historical OHLCV data from Yahoo Finance |
| `pandas` | ≥1.5 | Data manipulation and windowing |
| `numpy` | ≥1.23 | Numerical operations and array manipulation |
| `statsmodels` | ≥0.13 | ADF test, ACF/PACF plots, ARIMA model |
| `scikit-learn` | ≥1.1 | `StandardScaler` for feature normalisation |
| `tensorflow` / `keras` | ≥2.12 | Dense and LSTM neural network models |
| `matplotlib` | ≥3.6 | Visualisation of price series and predictions |

---

## Project Structure

```
netflix-stock-prediction/
│
├── netflix.ipynb              # Main notebook — all models and analysis
├── README.md                  # This file
│
└── (v2 — coming soon)
    └── netflix_returns.ipynb  # Absolute return prediction with stationary targets
```

---

## References

- Fama, E. F. (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work.* Journal of Finance, 25(2), 383–417.
- Malkiel, B. G. (1973). *A Random Walk Down Wall Street.* W. W. Norton & Company.
- Box, G. E. P., Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control.* Holden-Day.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.

---

*This project is for educational and research purposes only. Nothing in this repository constitutes financial advice.*
