# Netflix (NFLX) Stock Forecasting — v1 & v2

> **A two-phase time-series forecasting study on Netflix stock (2002–2025). v1 uses classical statistical models and deep learning on raw prices, empirically validating the Random Walk Theory and the Efficient Market Hypothesis (EMH) — proving that raw prices are fundamentally unpredictable. v2 addresses this by shifting the prediction target to absolute returns (daily price differences), a stationary series, and extends the model zoo with GRU networks and a leakage-free multivariate pipeline.**

---

## Table of Contents

1. [Why Netflix?](#why-netflix)
2. [Project Overview](#project-overview)
3. [Dataset](#dataset)
4. [Stationarity Analysis](#stationarity-analysis)
5. [Baseline Model — Naïve Forecast](#baseline-model--naïve-forecast)
6. [ARIMA Model](#arima-model)
7. [Deep Learning Models](#deep-learning-models)
   - [Univariate Dense Networks (Window-based)](#univariate-dense-networks-window-based)
   - [Univariate LSTM Networks](#univariate-lstm-networks)
   - [Multivariate Model with Technical Indicators](#multivariate-model-with-technical-indicators)
8. [v2 — Absolute Returns Prediction](#v2--absolute-returns-prediction)
   - [Target Transformation](#target-transformation)
   - [Naïve Baseline on Returns](#naïve-baseline-on-returns)
   - [Univariate Dense Networks (Returns)](#univariate-dense-networks-returns)
   - [Univariate LSTM Networks (Returns)](#univariate-lstm-networks-returns)
   - [Multivariate Models on Returns](#multivariate-models-on-returns)
   - [GRU Network](#gru-network)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Results Summary](#results-summary)
11. [Key Finding: Random Walk Theory & EMH](#key-finding-random-walk-theory--emh)
12. [Why v1 Fails — and What v2 Fixes](#why-v1-fails--and-what-v2-fixes)
13. [Model Persistence](#model-persistence)
14. [Tech Stack](#tech-stack)
15. [Project Structure](#project-structure)
16. [References](#references)

---

## Why Netflix?

Netflix (NFLX) is one of the most analytically interesting equities available for a time-series forecasting study, for several reasons.

**Extreme price range and regime changes.** NFLX went from a post-IPO price of around $1 (split-adjusted) in 2002 to an all-time high of ~$700 in November 2021, followed by a collapse of over 75% through 2022, and a subsequent recovery. This single ticker encapsulates bull runs, speculative bubbles, macro-driven crashes, and mean reversion — making it a far richer test bed than a blue-chip stock with a smooth upward drift.

**Volatility that challenges models.** Netflix's price is notoriously sensitive to earnings reports, subscriber growth numbers, and competitive dynamics (the entry of Disney+, HBO Max, etc.). This event-driven volatility makes the series highly non-stationary and difficult to model, which is precisely the property this project sets out to demonstrate and then address.

**Sector significance.** Netflix is the founding member of the FAANG group and is widely regarded as the stock that defined the streaming era. Its price trajectory reflects broader narratives in technology and consumer behaviour — growth-at-all-costs investing through the 2010s, the pandemic-era demand surge, and the post-2021 valuation reset. A model that can navigate NFLX's history must generalise across genuinely different market regimes.

**Data quality and availability.** Twenty-three years of clean, split-adjusted OHLCV data are freely available via Yahoo Finance, making NFLX an ideal choice for academic and portfolio projects alike.

In short: NFLX is a hard stock to predict, and that is the point. A model that fails here fails informatively — and a model that eventually succeeds here would be meaningful.

---

## Project Overview

This project attempts to forecast Netflix's daily closing stock price (v1) and daily absolute returns (v2) using historical price data from May 2002 to December 2025 (approximately 5,940 trading days). A range of approaches is implemented — from a simple naïve forecast to multi-layer stacked LSTMs and GRUs with multiple technical indicators — to rigorously test whether any ML model can beat a trivial baseline.

**v1** (`netflix_raw_prices.ipynb`) targets raw closing prices and arrives at a **negative** central conclusion: no model significantly outperforms the naïve forecast. This is not a failure of implementation — it is a meaningful empirical result that directly supports the **Random Walk Theory** and the **Efficient Market Hypothesis (EMH)**.

**v2** (`netflix_returns.ipynb`) corrects the root cause of v1's failure by predicting **absolute returns** — a stationary target — and extends the model suite with a GRU architecture and a leakage-free multivariate pipeline where technical indicators are computed separately for the train and test splits.

---

## Dataset

- **Source:** Yahoo Finance via the `yfinance` Python library
- **Ticker:** `NFLX`
- **Date Range:** 2002-05-23 to 2025-12-30
- **Total Observations:** ~5,940 trading days
- **Target Variable (v1):** Daily adjusted closing price (`Close`)
- **Target Variable (v2):** Absolute daily return — `returns(t) = Close(t) − Close(t−1)`
- **Train/Test Split:** 80% train / 20% test

**Why this split ratio?** An 80/20 split is the standard for time-series data where the model must be evaluated on unseen *future* data. Crucially, no shuffling is performed — temporal order is strictly preserved to prevent look-ahead bias, which would artificially inflate model performance.

**v2 split discipline:** In v2, the 80/20 split is applied to raw prices *before* computing returns. This ensures that `y1_returns = diff(y1)` and `y2_returns = diff(y2)` — there is no cross-contamination of the boundary price between the two splits, which would constitute a subtle form of data leakage.

---

## Stationarity Analysis

Before modelling, the statistical properties of the series are examined using the **Augmented Dickey-Fuller (ADF) test**.

### Raw Price Series (Non-Stationary)

```
ADF Statistic:  2.615
p-value:        0.999
```

The p-value of ~1.0 is far above any significance threshold (0.05), meaning we **fail to reject the null hypothesis of a unit root**. The raw price series is non-stationary — its mean and variance change over time, making it unsuitable for direct modelling with most statistical and ML methods.

### First-Differenced Series / Absolute Returns (Stationary)

```
ADF Statistic:  -12.086
p-value:        2.18e-22
```

After first-order differencing, the p-value collapses to essentially zero. The series is now stationary — confirming that **price changes (not price levels)** are the appropriate quantity to model. This is the core motivation for v2.

**Why does non-stationarity matter?**
A non-stationary series violates the fundamental assumptions of ARIMA, standard regression, and neural network training alike. Models trained on non-stationary data learn the local trend rather than any true predictive pattern, resulting in a deceptively low training loss but poor generalisation.

---

## Baseline Model — Naïve Forecast

The naïve forecast predicts that tomorrow's price (or return) equals today's price (or return):

```
ŷ(t) = y(t - 1)
```

This is the canonical baseline for financial time series and is surprisingly hard to beat. Its MASE score of **~1.0** defines the unit of comparison for all other models — any model with MASE > 1 is *worse* than doing nothing.

**v1 — Naïve on raw prices:**

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

The lower AIC of ARIMA(1,1,1) confirms it as the better-fitting specification. A **walk-forward validation** strategy is used: the model is re-fitted at every step with an expanding history window rather than making multi-step forecasts from a fixed origin. This avoids compounding forecast errors and represents best practice for ARIMA in production.

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

**Training setup:** All dense models share the same training configuration — `Adam` optimiser, MAE loss, `EarlyStopping(patience=10)` on validation loss, and `ReduceLROnPlateau(patience=10, factor=0.2, min_lr=1e-5)` to decay the learning rate when progress stalls. `restore_best_weights=True` ensures the best checkpoint is retained.

**Key observations:**
- The 7-day window performs best among the three, suggesting any historical information beyond one week adds noise rather than signal for raw prices.
- All three models have MASE > 1, meaning they are **all worse than the naïve forecast** in absolute terms.
- The large discrepancy between training MAE (< 0.5) and test MAE (13+) is a hallmark of a model that has learned to memorise the training distribution's upward trend rather than any generalisable pattern — a direct consequence of training on a non-stationary series.

**Why `StandardScaler` is fitted only on training data:** The scaler is fit exclusively on the training set and then applied to the test set. This prevents data leakage — using future statistics to normalise past data — which would be a form of look-ahead bias.

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

This is the most sophisticated setup in v1, incorporating three technical indicators alongside raw price.

**Feature Engineering:**
- **EMA-20 (20-day Exponential Moving Average):** A trend-following indicator that weights recent prices more heavily, smoothing short-term noise. Captures the long-term trend direction.
- **EMA-5 (5-day Exponential Moving Average):** A short-term trend indicator that reacts quickly to recent price moves. Used alongside EMA-20 to capture near-term momentum shifts that the longer EMA would smooth over.
- **MACD Histogram (12/26/9 configuration):** Measures momentum by comparing short and long-term EMAs. The histogram captures the acceleration of price momentum, signalling potential trend changes.

Each of these four features (`Close`, `EMA-20`, `EMA-5`, `MACD Histogram`) is lagged by 7 periods, creating a `[7 × 4]` input matrix of 28 features per sample. Two separate `StandardScaler` instances are used — one for inputs and one for the target — to avoid scale contamination between features and labels.

**Multivariate LSTM (`model_3_lstm`):**
```
Lambda(reshape → [batch, 7, 4]) → LSTM(32, relu, return_sequences=True)
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

The multivariate Dense network (`model_4`) shows the best MAE of all deep learning models (1.11), yet its MASE of 1.177 still fails to beat the naïve baseline. The addition of EMA-20, EMA-5, and MACD provides some marginal improvement over univariate models, but these indicators are derived from the same non-stationary price series — not from any genuinely independent predictive signal.

---

## v2 — Absolute Returns Prediction

### Target Transformation

v2 replaces the raw closing price with **absolute returns** as the prediction target:

```
returns(t) = Close(t) − Close(t−1)
```

This is exactly the first-differencing operation whose stationarity is already confirmed in the v1 notebook (ADF p ≈ 2e-22). By shifting the target from price levels to price changes, the input distribution becomes stable across the entire training period — the model no longer has to generalise across a series that ranged from $1 to $694.

| Property | Raw Price | Absolute Return |
|----------|-----------|-----------------||
| Stationarity | ❌ Non-stationary | ✅ Stationary |
| Scale | Varies ($1 → $694) | Stable (centred near 0) |
| Economic meaning | Absolute price level | Dollar-denominated daily change |
| Suitable for ML | ❌ No | ✅ Yes |
| Additive over time | ❌ No | ✅ Yes |

**No `StandardScaler` in v2 univariate models:** Because absolute returns are already near-zero-centred and have a stable variance across time, the univariate models in v2 operate directly on raw return values without scaling. This simplifies the pipeline and eliminates the need for inverse transforms at evaluation time.

---

### Naïve Baseline on Returns

The naïve forecast on returns carries over the same logic: tomorrow's return equals today's return.

| Metric | Value |
|--------|-------|
| MAE    | 1.357 |
| RMSE   | 2.048 |
| MAPE   | — *(unreliable; returns are near zero)* |
| MASE   | **0.999** |

> **Note on MAPE:** MAPE is mathematically undefined (or astronomically large) when the true value is close to zero, which is common for daily returns. The raw value from the notebook (414,440) is meaningless and is therefore omitted. MASE remains the primary metric throughout v2.

---

### Univariate Dense Networks (Returns)

The same three window sizes (7, 10, 14 days) are evaluated on the returns series, using the identical `Dense(128, relu) → Dense(1)` architecture and the same training callbacks as v1.

| Model | Window Size | MAE | RMSE | MASE |
|-------|-------------|-----|------|------|
| Model 1 (Dense) | 7 days  | 0.961 | 1.477 | **0.709** |
| Model 2 (Dense) | 10 days | 0.964 | 1.476 | **0.712** |
| Model 3 (Dense) | 14 days | 0.983 | 1.492 | **0.724** |

All three beat the naïve baseline (MASE < 1), a direct result of switching to a stationary target. The 7-day window again performs best, consistent with v1's finding that week-level lookbacks are sufficient for daily NFLX data.

---

### Univariate LSTM Networks (Returns)

Two stacked LSTM variants are tested on the 7-day returns window, with the activation order *reversed* relative to v1 — here tanh is tested first and relu second. This is intentional: v1 demonstrated that tanh is pathological for raw prices, whereas v2 tests whether it recovers on a stationary target.

**Model `uni_lstm_1` — Tanh activation:**
```
Lambda(expand_dims) → LSTM(32, tanh, recurrent_dropout=0.2, return_sequences=True)
                    → LSTM(32, tanh, recurrent_dropout=0.2) → Dense(1)
```

**Model `uni_lstm_2` — ReLU activation:**
```
Lambda(expand_dims) → LSTM(32, relu, recurrent_dropout=0.2, return_sequences=True)
                    → LSTM(32, relu, recurrent_dropout=0.2) → Dense(1)
```

| Model | Activation | MAE | RMSE | MASE |
|-------|-----------|-----|------|------|
| LSTM 1 (uni_lstm_1) | Tanh | 0.943 | 1.462 | **0.696** |
| LSTM 2 (uni_lstm_2) | ReLU | 0.944 | 1.462 | **0.696** |

Both models beat the naïve baseline and perform virtually on par — a sharp contrast to v1 where both LSTMs catastrophically diverged (MASE 3.0 and 15.9). This confirms that the stationary target is the decisive fix. Both use `EarlyStopping(patience=10)` and `ReduceLROnPlateau(patience=10, factor=0.2, min_lr=1e-5)`. The ReLU variant (`model_uni_lstm_2`) is saved as a `.keras` artifact for downstream use.

---

### Multivariate Models on Returns

The multivariate pipeline in v2 introduces a critical architectural change over v1: **technical indicators are computed independently on the train and test splits** rather than on the full price series before splitting. This eliminates lookahead bias in the indicator computation itself, which was a subtle leakage vector in v1.

**Feature Engineering on Returns:**
- **EMA-20 of returns:** Tracks the smoothed medium-term trend of daily price changes.
- **EMA-5 of returns:** Captures short-term momentum in the return series.
- **MACD Histogram of returns (12/26/9):** Measures acceleration of momentum within the return series rather than the raw price series.

Each of these four features (`returns`, `EMA-20`, `EMA-5`, `MACD Histogram`) is lagged 7 periods, producing a `[7 × 4]` input of 28 features per sample — identical in shape to v1's multivariate setup.

**No StandardScaler on returns multivariate inputs or targets:** Since the return series is stationary and near-zero-centred, no feature scaling is applied. The model trains directly on raw return values and the `model_preds_uni` helper function is reused without any inverse transform.

Three multivariate models are evaluated:

**Multivariate Dense (`model_multi_1`):**
```
Dense(128, relu) → Dense(1)
```

**Multivariate LSTM (`model_multi_lstm`):**
```
Lambda(expand_dims) → LSTM(32, tanh, recurrent_dropout=0.2, return_sequences=True)
                    → LSTM(32, tanh, recurrent_dropout=0.2) → Dense(1)
```

**Multivariate GRU (`model_multi_gru`):**
```
Lambda(expand_dims) → GRU(32, tanh, recurrent_dropout=0.2, return_sequences=True)
                    → GRU(32, tanh, recurrent_dropout=0.2) → Dense(1)
```

| Model | MAE | RMSE | MASE | Beats Naïve? |
|-------|-----|------|------|:---:|
| Multivariate Dense | 0.960 | 1.471 | **0.709** | ✅ |
| Multivariate LSTM (Tanh) | 0.942 | 1.462 | **0.695** | ✅ |
| Multivariate GRU (Tanh) | 0.942 | 1.463 | **0.695** | ✅ |

The multivariate LSTM and GRU are the best-performing models overall, with MASE of 0.695 — roughly 30% better than naïve. The multivariate Dense model (0.709) also beats naïve, though by a smaller margin than the recurrent models.

---

### GRU Network

The **Gated Recurrent Unit (GRU)** is introduced in v2 as an alternative to LSTM. GRUs use two gates (reset and update) instead of LSTM's three (input, forget, output), making them computationally lighter while achieving comparable performance on many sequence tasks. On a stationary returns series with relatively low autocorrelation, the simpler gating structure of GRU may be less prone to over-fitting than LSTM. The best-performing multivariate model (`model_multi_gru`) is saved as a `.keras` artifact.

---

## Evaluation Metrics

Four complementary metrics are used to evaluate all models:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | `mean(|y_true - y_pred|)` | Average absolute error in raw units (USD for prices, USD for returns). Scale-dependent. |
| **RMSE** | `sqrt(mean((y_true - y_pred)²))` | Penalises large errors more heavily than MAE. Sensitive to outliers. |
| **MAPE** | `mean(|y_true - y_pred| / y_true) × 100` | Percentage error — scale-invariant but undefined at zero and biased for small values. Less reliable for returns, which can be near zero. |
| **MASE** | `MAE / MAE_naïve` | **The primary metric.** A MASE < 1 beats the naïve baseline; MASE = 1 equals it; MASE > 1 is worse. Scale-invariant and directly interpretable. |

**Why MASE is the primary metric:** In financial forecasting, the question is not "how large is the error in dollars?" but "does this model add any value over the simplest possible strategy?" MASE directly answers this by normalising error against the naïve forecast.

**Note on MAPE for returns:** MAPE becomes unreliable when the true value is near zero (as daily returns frequently are). It is retained for comparability with v1 but should be interpreted cautiously in the v2 context.

---

## Results Summary

### v1 — Raw Prices

| Model | Window / Config | MAE | MASE | Beats Naïve? |
|-------|----------------|-----|------|:---:|
| Naïve Forecast | — | 0.939 | 1.000 | — (baseline) |
| ARIMA(1,1,1) | Walk-forward | 0.945 | 1.007 | ❌ |
| Dense (Univariate) | 7-day | 13.75 | 1.030 | ❌ |
| Dense (Univariate) | 10-day | 14.13 | 1.060 | ❌ |
| Dense (Univariate) | 14-day | 14.29 | 1.070 | ❌ |
| LSTM (ReLU) | 7-day | 40.05 | 3.001 | ❌ |
| LSTM (Tanh) | 7-day | 212.32 | 15.91 | ❌ |
| Multivariate LSTM | 7-day × 4 features | 2.32 | 2.470 | ❌ |
| Multivariate Dense | 7-day × 4 features | 1.11 | **1.177** | ❌ |

**No model in v1 beats the naïve forecast on MASE.**

### v2 — Absolute Returns

*(Exact numeric results depend on execution environment. The key architectural and methodological improvements are listed below.)*

| Model | Window / Config | MAE | MASE | Beats Naïve? |
|-------|----------------|-----|------|:---:|
| Naïve Forecast | — | 1.357 | 1.000 | — (baseline) |
| Dense (Univariate) | 7-day | 0.961 | 0.709 | ✅ |
| Dense (Univariate) | 10-day | 0.964 | 0.712 | ✅ |
| Dense (Univariate) | 14-day | 0.983 | 0.724 | ✅ |
| LSTM (Tanh) | 7-day | 0.943 | 0.696 | ✅ |
| LSTM (ReLU) | 7-day | 0.944 | 0.696 | ✅ |
| Multivariate Dense | 7-day × 4 features | 0.960 | 0.709 | ✅ |
| Multivariate LSTM (Tanh) | 7-day × 4 features | 0.942 | **0.695** | ✅ |
| **Multivariate GRU (Tanh)** | 7-day × 4 features | **0.942** | **0.695** | ✅ |

**Every model in v2 beats the naïve forecast. The best models (Multivariate LSTM and GRU) achieve a MASE of 0.695 — approximately 30% better than naïve.**

---

## Key Finding: Random Walk Theory & EMH

The failure of every model in v1 is not a bug — it is the expected result according to two foundational theories in financial economics.

### Random Walk Theory

The Random Walk Hypothesis (Malkiel, 1973) states that stock price changes are serially independent — each price movement is statistically unrelated to past movements. Formally:

```
P(t) = P(t-1) + ε(t),    where ε(t) ~ white noise
```

If this holds, then the best prediction of tomorrow's price is simply today's price — exactly what the naïve model does. The ADF test results in this project directly corroborate this: after first differencing, the series resembles white noise with no significant autocorrelation at any lag (as confirmed by the ACF/PACF plots). There is no exploitable pattern.

### Efficient Market Hypothesis (EMH)

The EMH (Fama, 1970) asserts that asset prices at any time fully reflect all available information. In its **weak form**, it specifically claims that no trading strategy based solely on historical price data can consistently generate excess returns — because all such information is already priced in.

Every model in this project uses only historical price and price-derived features (EMA-5, EMA-20, MACD). The fact that none outperforms the naïve forecast in v1 is direct empirical evidence in support of the weak-form EMH for NFLX over this period.

**In short:** v1 does not fail to predict stock prices. It *succeeds* in demonstrating that raw stock prices cannot be predicted from their own history, which is precisely what theory predicts.

---

## Why v1 Fails — and What v2 Fixes

### The Root Cause: Non-Stationarity

The core technical problem is that raw closing prices are **non-stationary**. They have a time-varying mean (the long-term upward trend of NFLX from ~$1 to ~$700) and a time-varying variance (volatility clusters). Models trained on non-stationary data do not learn generalisable patterns — they learn the local trend of the training set and extrapolate it, which fails on the test set.

This is why models show low training loss but drastically higher test loss: the distributional shift between the pre-2021 training data and the post-2021 test data (which includes NFLX's peak at ~$700 and its 2022 crash) is enormous.

### The v2 Fix: Absolute Returns + Leakage-Free Multivariate Pipeline

v2 applies two corrections simultaneously:

**1. Stationary target:** Raw prices are replaced with absolute returns — the first-differenced series confirmed stationary by the ADF test. The input distribution is now stable across the full training horizon; the model is no longer forced to extrapolate across a price range spanning two orders of magnitude.

**2. Leakage-free indicator computation:** In v1, EMA and MACD are computed on the full price series before splitting, meaning the training-set indicators technically incorporate forward-looking statistics from the normalisation of the joint series. In v2, indicators are computed separately on `netflix_train_price_df` and `netflix_test_price_df` — eliminating this subtle lookahead bias.

**3. No input scaling for multivariate returns models:** Because the return series and its derived indicators are already near-zero-centred, `StandardScaler` is omitted from the multivariate v2 pipeline entirely. This removes one source of complexity and one potential leakage point.

---

## Model Persistence

Trained models and scalers are saved for downstream inference and deployment:

**v1 artifacts (saved via `joblib` and Keras):**
- `Models/scaler_prices.pkl` — univariate `StandardScaler`
- `Models/scaler_multi_input_prices.pkl` — multivariate input scaler
- `Models/scaler_multi_target_prices.pkl` — multivariate target scaler
- `Models/dense_model_uni_prices.keras` — best univariate dense model (7-day)

**v2 artifacts (saved via Keras):**
- `Models/lstm_uni_returns.keras` — best univariate LSTM on returns (`model_uni_lstm_2`, ReLU)
- `Models/gru_multi_returns.keras` — best multivariate GRU on returns

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `yfinance` | latest | Historical OHLCV data from Yahoo Finance |
| `pandas` | ≥1.5 | Data manipulation and feature engineering |
| `numpy` | ≥1.23 | Numerical operations and array manipulation |
| `statsmodels` | ≥0.13 | ADF test, ACF/PACF plots, ARIMA model |
| `scikit-learn` | ≥1.1 | `StandardScaler` for feature normalisation (v1) |
| `tensorflow` / `keras` | ≥2.12 | Dense, LSTM, and GRU neural network models |
| `matplotlib` | ≥3.6 | Visualisation of price series, returns, and predictions |
| `joblib` | latest | Model and scaler persistence |

---

## Project Structure

```
netflix-stock-prediction/
│
├── netflix_raw_prices.ipynb   # v1 — raw price prediction; proves EMH empirically
├── netflix_returns.ipynb      # v2 — absolute returns prediction; stationary target
├── README.md                  # This file
│
└── Models/
    ├── scaler_prices.pkl                  # v1 univariate scaler
    ├── scaler_multi_input_prices.pkl      # v1 multivariate input scaler
    ├── scaler_multi_target_prices.pkl     # v1 multivariate target scaler
    ├── dense_model_uni_prices.keras       # v1 best univariate dense (7-day window)
    ├── lstm_uni_returns.keras             # v2 best univariate LSTM (relu, 7-day)
    └── gru_multi_returns.keras            # v2 best multivariate GRU
```

---

## References

- Fama, E. F. (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work.* Journal of Finance, 25(2), 383–417.
- Malkiel, B. G. (1973). *A Random Walk Down Wall Street.* W. W. Norton & Company.
- Box, G. E. P., Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control.* Holden-Day.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
- Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.* arXiv:1406.1078. *(Introduces the GRU architecture.)*

---

*This project is for educational and research purposes only. Nothing in this repository constitutes financial advice.*
