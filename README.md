# Netflix (NFLX) Stock Forecasting — Raw Prices and Absolute Returns

> A two-phase time-series forecasting study on Netflix stock (2002–2025). The raw prices notebook uses classical statistical models and deep learning on raw closing prices, empirically validating the Random Walk Theory and the Efficient Market Hypothesis — proving that raw prices are fundamentally unpredictable from their own history. The returns notebook addresses this limitation by shifting the prediction target to absolute daily returns, a stationary series, and extends the model suite with GRU networks and a leakage-free multivariate pipeline. Together, the two notebooks form a complete, self-contained study in why target choice is the most consequential modelling decision in financial time-series forecasting.

---

## Table of Contents

1. [Why Netflix?](#why-netflix)
2. [Project Overview](#project-overview)
3. [Dataset](#dataset)
4. [Stationarity Analysis](#stationarity-analysis)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Baseline Model — Naive Forecast](#baseline-model--naive-forecast)
7. [ARIMA Model](#arima-model)
8. [Raw Prices Notebook — Deep Learning Models](#raw-prices-notebook--deep-learning-models)
   - [Univariate Dense Networks](#univariate-dense-networks)
   - [Univariate LSTM Networks](#univariate-lstm-networks)
   - [Multivariate Model with Technical Indicators](#multivariate-model-with-technical-indicators)
9. [Returns Notebook — Absolute Returns Prediction](#returns-notebook--absolute-returns-prediction)
   - [Target Transformation](#target-transformation)
   - [Naive Baseline on Returns](#naive-baseline-on-returns)
   - [Univariate Dense Networks (Returns)](#univariate-dense-networks-returns)
   - [Univariate LSTM Networks (Returns) — Tanh Vindicated](#univariate-lstm-networks-returns--tanh-vindicated)
   - [Multivariate Models on Returns](#multivariate-models-on-returns)
   - [GRU Network](#gru-network)
10. [Cross-Cutting Observations](#cross-cutting-observations)
    - [Why Dense and RNN Models Perform Similarly](#why-dense-and-rnn-models-perform-similarly)
    - [Why Multivariate Offers Only Marginal Gains](#why-multivariate-offers-only-marginal-gains)
    - [The MAPE Problem in Returns Forecasting](#the-mape-problem-in-returns-forecasting)
11. [Results Summary](#results-summary)
12. [Key Finding: Random Walk Theory and EMH](#key-finding-random-walk-theory-and-emh)
13. [Why Raw Prices Fail — and What the Returns Notebook Fixes](#why-raw-prices-fail--and-what-the-returns-notebook-fixes)
14. [Overall Impact](#overall-impact)
15. [Model Persistence](#model-persistence)
16. [Tech Stack](#tech-stack)
17. [Project Structure](#project-structure)
18. [References](#references)

---

## Why Netflix?

Netflix (NFLX) is one of the most analytically interesting equities available for a time-series forecasting study, for several reasons.

**Extreme price range and regime changes.** NFLX went from a post-IPO price of around $1 (split-adjusted) in 2002 to an all-time high of approximately $700 in November 2021, followed by a collapse of over 75% through 2022, and a subsequent recovery. This single ticker encapsulates bull runs, speculative bubbles, macro-driven crashes, and mean reversion — making it a far richer test bed than a blue-chip stock with a smooth upward drift.

**Volatility that challenges models.** Netflix's price is notoriously sensitive to earnings reports, subscriber growth numbers, and competitive dynamics (the entry of Disney+, HBO Max, etc.). This event-driven volatility makes the series highly non-stationary and difficult to model, which is precisely the property this project sets out to demonstrate and then address.

**Sector significance.** Netflix is the founding member of the FAANG group and is widely regarded as the stock that defined the streaming era. Its price trajectory reflects broader narratives in technology and consumer behaviour — growth-at-all-costs investing through the 2010s, the pandemic-era demand surge, and the post-2021 valuation reset. A model that can navigate NFLX's history must generalise across genuinely different market regimes.

**Data quality and availability.** Twenty-three years of clean, split-adjusted OHLCV data are freely available via Yahoo Finance, making NFLX an ideal choice for academic and portfolio projects alike.

In short: NFLX is a hard stock to predict, and that is the point. A model that fails here fails informatively — and a model that eventually succeeds here would be meaningful.

---

## Project Overview

This project attempts to forecast Netflix's daily closing stock price (raw prices notebook) and daily absolute returns (returns notebook) using historical price data from May 2002 to December 2025 — approximately 5,940 trading days. A range of approaches is implemented, from a simple naive forecast to multi-layer stacked LSTMs and GRUs with multiple technical indicators, to rigorously test whether any model can beat a trivial baseline.

**Raw prices notebook** (`netflix_raw_prices.ipynb`) targets raw closing prices and arrives at a negative central conclusion: no model significantly outperforms the naive forecast. This is not a failure of implementation — it is a meaningful empirical result that directly supports the Random Walk Theory and the Efficient Market Hypothesis (EMH). The notebook serves as a formal, empirical proof-of-concept for those theories.

**Returns notebook** (`netflix_returns.ipynb`) corrects the root cause of the raw prices failure by predicting absolute returns — a stationary target — and extends the model suite with a GRU architecture and a leakage-free multivariate pipeline where technical indicators are computed separately for the train and test splits.

---

## Dataset

- **Source:** Yahoo Finance via the `yfinance` Python library
- **Ticker:** `NFLX`
- **Date Range:** 2002-05-23 to 2025-12-30
- **Total Observations:** approximately 5,940 trading days
- **Target Variable (raw prices notebook):** Daily adjusted closing price (`Close`)
- **Target Variable (returns notebook):** Absolute daily return — `returns(t) = Close(t) - Close(t-1)`
- **Train/Test Split:** 80% train / 20% test

**Why this split ratio?** An 80/20 split is the standard for time-series data where the model must be evaluated on unseen future data. No shuffling is performed — temporal order is strictly preserved to prevent look-ahead bias, which would artificially inflate model performance.

**Split discipline in the returns notebook:** The 80/20 split is applied to raw prices before computing returns. This ensures that `y1_returns = diff(y1)` and `y2_returns = diff(y2)` — there is no cross-contamination of the boundary price between the two splits, which would constitute a subtle form of data leakage.

---

## Stationarity Analysis

Before modelling, the statistical properties of the series are examined using the **Augmented Dickey-Fuller (ADF) test**. This test checks for the presence of a unit root — a property that makes a series non-stationary — by testing the null hypothesis that a unit root exists. Rejecting the null (low p-value) means the series is stationary.

### Raw Price Series (Non-Stationary)

```
ADF Statistic:  2.615
p-value:        0.999
```

The p-value of approximately 1.0 is far above any significance threshold (0.05), meaning we fail to reject the null hypothesis of a unit root. The raw price series is non-stationary — its mean and variance change over time, making it unsuitable for direct modelling with most statistical and ML methods.

### First-Differenced Series / Absolute Returns (Stationary)

```
ADF Statistic:  -12.086
p-value:        2.18e-22
```

After first-order differencing, the p-value collapses to essentially zero. The series is now stationary, confirming that price changes (not price levels) are the appropriate quantity to model. This result directly motivates the returns notebook.

**Why does non-stationarity matter?** A non-stationary series violates the fundamental assumptions of ARIMA, standard regression, and neural network training alike. Models trained on non-stationary data learn the local trend of the training period rather than any true predictive pattern. This produces deceptively low training loss but fails badly on the test set, because the distributional properties of the series have shifted.

---

## Evaluation Metrics

Four complementary metrics are used to evaluate all models throughout this project. Understanding each metric — and when it is reliable — is essential for interpreting the results correctly.

**MAE — Mean Absolute Error**

```
MAE = mean(|y_true - y_pred|)
```

The average absolute error in raw units (USD for prices, USD for returns). MAE is intuitive and easy to interpret but is scale-dependent, meaning it cannot be compared across different targets (e.g., raw prices vs. returns) without normalisation.

**RMSE — Root Mean Squared Error**

```
RMSE = sqrt(mean((y_true - y_pred)^2))
```

Penalises large errors more heavily than MAE because errors are squared before averaging. RMSE is useful for detecting whether a model is making occasional catastrophic predictions, even if its average error looks acceptable. Like MAE, it is scale-dependent.

**MAPE — Mean Absolute Percentage Error**

```
MAPE = mean(|y_true - y_pred| / |y_true|) x 100
```

Expresses error as a percentage of the true value, making it scale-invariant and useful for comparing models across different price regimes. However, MAPE has a critical flaw: it becomes numerically unstable — and effectively meaningless — when the true value is near zero. For raw price forecasting (where prices are always positive and far from zero), MAPE is a valid supplementary metric. For returns forecasting (where daily returns frequently pass through zero), MAPE explodes to astronomically large values. These inflated values are reported in full throughout this document for completeness, but they carry no interpretive weight in the returns setting. MASE is the authoritative metric for all comparisons.

**Why MAPE is still included despite its flaw:** Even in the returns setting, retaining MAPE serves two purposes. First, it maintains a consistent metric table structure, making it straightforward to compare entries across both notebooks without tracking which metrics apply where. Second, MAPE can serve as a conditional diagnostic — if two models have similar MAE and MASE but very different MAPE values, it signals that one model is performing worse specifically on near-zero return days, a type of conditional failure that MASE would average out. In this project, MAPE values across models in the returns setting are all similarly inflated, which confirms the issue is structural (denominator near zero) rather than model-specific.

**MASE — Mean Absolute Scaled Error (Primary Metric)**

```
MASE = MAE_model / MAE_naive
```

MASE normalises a model's MAE against the MAE of the naive forecast (which predicts tomorrow equals today). A MASE below 1.0 means the model outperforms doing nothing. A MASE above 1.0 means it is worse than doing nothing. MASE is scale-invariant, interpretable, and directly answers the central question of any forecasting study: does this model add value over the simplest possible baseline?

MASE is the primary metric throughout this project for all ranking and conclusion purposes.

---

## Baseline Model — Naive Forecast

The naive forecast predicts that tomorrow's price (or return) equals today's price (or return):

```
y_hat(t) = y(t - 1)
```

This is the canonical baseline for financial time series and is surprisingly hard to beat. Its MASE score of approximately 1.0 defines the unit of comparison for all other models — any model with MASE greater than 1 is worse than doing nothing.

**Raw prices notebook — Naive on closing prices:**

| Metric | Value |
|--------|-------|
| MAE    | 0.939 |
| RMSE   | 1.458 |
| MAPE   | 1.764% |
| MASE   | 0.999 |

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
| MASE   | 1.007 |

ARIMA(1,1,1) is marginally worse than the naive baseline. This is consistent with the Random Walk Hypothesis: the first-differenced stock price resembles white noise with no exploitable autocorrelation structure. The model finds nothing useful to learn.

---

## Raw Prices Notebook — Deep Learning Models

### Univariate Dense Networks

A sliding-window approach converts the 1D price series into supervised learning inputs. Three window sizes are evaluated to find the optimal lookback horizon.

**Architecture:**
```
Dense(128, activation='relu') -> Dense(1)
```

**Training setup:** All models use the `Adam` optimiser, MAE loss, `EarlyStopping(patience=10)` on validation loss, and `ReduceLROnPlateau(patience=10, factor=0.2, min_lr=1e-5)`. `restore_best_weights=True` ensures the best checkpoint is retained.

| Model | Window Size | MAE | RMSE | MAPE | MASE |
|-------|-------------|-----|------|------|------|
| Model 1 (Dense) | 7 days  | 13.75 | 21.14 | 1.79% | 1.030 |
| Model 2 (Dense) | 10 days | 14.13 | 21.54 | 1.87% | 1.060 |
| Model 3 (Dense) | 14 days | 14.29 | 21.66 | 1.88% | 1.070 |

All three models are worse than the naive forecast. The 7-day window performs best, suggesting historical information beyond one week adds noise rather than signal when the target is a non-stationary price level. The large discrepancy between training MAE (below 0.5) and test MAE (above 13) is a hallmark of distributional shift — the model memorises the upward trend of the training period and then extrapolates it into the test period where the distribution has radically changed.

**Why `StandardScaler` is fitted only on training data:** The scaler is fit exclusively on the training set and then applied to the test set. Fitting on the full dataset would incorporate future statistics into the normalisation of past data — a form of look-ahead bias.

---

### Univariate LSTM Networks

Two stacked LSTM architectures are tested on the 7-day window (best from the Dense experiment):

**Model `uni_lstm1` — ReLU activation:**
```
Lambda(expand_dims) -> LSTM(32, relu, recurrent_dropout=0.2, return_sequences=True)
                    -> LSTM(32, relu, recurrent_dropout=0.2) -> Dense(1)
```

**Model `uni_lstm2` — Tanh activation:**
```
Lambda(expand_dims) -> LSTM(32, tanh, recurrent_dropout=0.2, return_sequences=True)
                    -> LSTM(32, tanh, recurrent_dropout=0.2) -> Dense(1)
```

| Model | Activation | MAE | RMSE | MAPE | MASE |
|-------|-----------|-----|------|------|------|
| LSTM 1 | ReLU | 40.05 | 51.17 | 5.08% | 3.001 |
| LSTM 2 | Tanh | 212.32 | 376.30 | 15.84% | 15.91 |

Both models fail catastrophically — the ReLU variant is three times worse than naive, and the tanh variant nearly sixteen times worse. The reasons are distinct for each.

**Why ReLU fails in LSTMs:** LSTMs were designed with `tanh` and `sigmoid` gate activations that naturally bound internal state values to a finite range. Substituting `relu` for the recurrent activation disrupts this gating mechanism — unbounded activations can compound through time steps and cause the internal state to explode. A MASE of 3.0 reflects exactly this instability.

**Why tanh fails so severely in the raw prices notebook:** The tanh LSTM's catastrophic performance (MASE 15.91) is not a verdict against tanh as an activation — tanh is theoretically the correct choice for LSTM gates. The failure is a symptom of the non-stationary target. The raw NFLX price series has a strong upward trend with a time-varying mean: it went from under $1 to over $700. When tanh — a bounded activation with output range (-1, 1) — is applied to the hidden state of a network trying to track this unbounded, trending signal, the gating mechanism saturates. The gradients vanish, the network's memory collapses, and it produces predictions that diverge wildly from the test set. In short, tanh is not the problem; the non-stationary, trending input is the problem, and tanh makes the network more sensitive to it than ReLU because of its bounded nature. This is a diagnostic about the data, not a verdict about the architecture.

`recurrent_dropout=0.2` is used as a regularisation technique specific to RNNs. Unlike standard dropout, recurrent dropout applies the same mask at each time step, preserving the temporal flow of gradients.

---

### Multivariate Model with Technical Indicators

The most sophisticated setup in the raw prices notebook incorporates three technical indicators alongside raw price.

**Feature Engineering:**
- **EMA-20 (20-day Exponential Moving Average):** A trend-following indicator that weights recent prices more heavily. Captures the long-term trend direction.
- **EMA-5 (5-day Exponential Moving Average):** A short-term trend indicator that reacts quickly to recent price moves. Captures near-term momentum shifts that the longer EMA would smooth over.
- **MACD Histogram (12/26/9 configuration):** Measures momentum by comparing short and long-term EMAs. The histogram captures the acceleration of price momentum, signalling potential trend changes.

Each of the four features (`Close`, `EMA-20`, `EMA-5`, `MACD Histogram`) is lagged by 7 periods, creating a `[7 x 4]` input matrix of 28 features per sample. Two separate `StandardScaler` instances are used — one for inputs and one for the target — to avoid scale contamination.

**Multivariate LSTM (`model_3_lstm`):**
```
Lambda(reshape -> [batch, 7, 4]) -> LSTM(32, relu, return_sequences=True)
                                 -> LSTM(32, relu) -> Dense(1)
```

**Multivariate Dense (`model_4`):**
```
Dense(128, relu) -> Dense(1)
```

| Model | MAE | RMSE | MAPE | MASE |
|-------|-----|------|------|------|
| Multivariate LSTM | 2.32 | 3.47 | 4.45% | 2.470 |
| Multivariate Dense | 1.11 | 1.64 | 2.17% | 1.177 |

The multivariate Dense model achieves the best MAE of all deep learning models in this notebook (1.11), yet its MASE of 1.177 still fails to beat the naive baseline. EMA and MACD are derived entirely from the same non-stationary price series — they carry no genuinely independent predictive signal, only redundant transformations of the same trending data.

---

## Returns Notebook — Absolute Returns Prediction

### Target Transformation

The returns notebook replaces the raw closing price with absolute returns as the prediction target:

```
returns(t) = Close(t) - Close(t-1)
```

This is exactly the first-differencing operation confirmed stationary by the ADF test (p approximately 2e-22). By shifting the target from price levels to price changes, the input distribution becomes stable across the entire training period — the model no longer has to generalise across a series that ranged from $1 to $694.

| Property | Raw Price | Absolute Return |
|----------|-----------|----------------|
| Stationarity | Non-stationary | Stationary |
| Scale | Varies ($1 to $694) | Stable (centred near 0) |
| Economic meaning | Absolute price level | Dollar-denominated daily change |
| Suitable for ML | No | Yes |
| Additive over time | No | Yes |

**No `StandardScaler` in returns notebook univariate models:** Because absolute returns are already near-zero-centred and have a stable variance across time, the univariate models in the returns notebook operate directly on raw return values without scaling. This simplifies the pipeline and eliminates any inverse-transform step at evaluation time.

---

### Naive Baseline on Returns

The naive forecast on returns carries over the same logic: tomorrow's return equals today's return.

| Metric | Value |
|--------|-------|
| MAE    | 1.357 |
| RMSE   | 2.048 |
| MAPE   | 414,440% (structurally inflated — see section on MAPE) |
| MASE   | 0.999 |

---

### Univariate Dense Networks (Returns)

The same three window sizes (7, 10, 14 days) are evaluated on the returns series, using the identical `Dense(128, relu) -> Dense(1)` architecture and the same training callbacks.

| Model | Window Size | MAE | RMSE | MASE |
|-------|-------------|-----|------|------|
| Model 1 (Dense) | 7 days  | 0.961 | 1.477 | 0.709 |
| Model 2 (Dense) | 10 days | 0.964 | 1.476 | 0.712 |
| Model 3 (Dense) | 14 days | 0.983 | 1.492 | 0.724 |

All three beat the naive baseline — a direct result of switching to a stationary target. The 7-day window again performs best. MAPE values for these models are in the tens of thousands of percent and are omitted from this table for the same structural reason as the naive baseline.

---

### Univariate LSTM Networks (Returns) — Tanh Vindicated

Two stacked LSTM variants are tested on the 7-day returns window:

**Model `uni_lstm_1` — Tanh activation:**
```
Lambda(expand_dims) -> LSTM(32, tanh, recurrent_dropout=0.2, return_sequences=True)
                    -> LSTM(32, tanh, recurrent_dropout=0.2) -> Dense(1)
```

**Model `uni_lstm_2` — ReLU activation:**
```
Lambda(expand_dims) -> LSTM(32, relu, recurrent_dropout=0.2, return_sequences=True)
                    -> LSTM(32, relu, recurrent_dropout=0.2) -> Dense(1)
```

| Model | Activation | MAE | RMSE | MASE |
|-------|-----------|-----|------|------|
| LSTM 1 (uni_lstm_1) | Tanh | 0.943 | 1.462 | 0.696 |
| LSTM 2 (uni_lstm_2) | ReLU | 0.944 | 1.462 | 0.696 |

Both models beat the naive baseline and perform virtually on par — a sharp contrast to the raw prices notebook where tanh produced a MASE of 15.91 and ReLU produced 3.001.

**Why tanh recovers in the returns notebook:** In the raw prices notebook, the tanh LSTM collapsed because it was trying to track an unbounded, trending signal using bounded activations. The hidden state saturated, gradients vanished, and the network lost its memory entirely. In the returns notebook, the target is stationary and near-zero-centred. The values the LSTM needs to track — daily dollar changes fluctuating around zero — are well within the natural operating range of tanh's (-1, 1) output. The gating mechanism can function as intended: the forget gate learns what to discard, the input gate learns what to absorb, and gradients flow stably across time steps. The non-stationarity of raw prices was masking what tanh is actually well-suited to do. The returns notebook rehabilitates it fully.

The Tanh variant (`model_uni_lstm_1`) is saved as a `.keras` artifact for downstream use.

---

### Multivariate Models on Returns

The multivariate pipeline in the returns notebook introduces a critical correction over the raw prices version: **technical indicators are computed independently on the train and test splits** rather than on the full price series before splitting. In the raw prices notebook, EMA and MACD were computed over the entire series, meaning the training-set indicators subtly embedded information from the joint distribution that included future observations. In the returns notebook this is corrected by computing indicators separately on the training and test dataframes.

**Feature Engineering on Returns:**
- **EMA-20 of returns:** Tracks the smoothed medium-term trend of daily price changes.
- **EMA-5 of returns:** Captures short-term momentum in the return series.
- **MACD Histogram of returns (12/26/9):** Measures momentum acceleration within the return series rather than the raw price series.

Each of the four features (`returns`, `EMA-20`, `EMA-5`, `MACD Histogram`) is lagged 7 periods, producing a `[7 x 4]` input of 28 features per sample — identical in shape to the raw prices multivariate setup. No StandardScaler is applied since the return series is stationary and near-zero-centred.

**Multivariate Dense (`model_multi_1`):**
```
Dense(128, relu) -> Dense(1)
```

**Multivariate LSTM (`model_multi_lstm`):**
```
Lambda(expand_dims) -> LSTM(32, tanh, recurrent_dropout=0.2, return_sequences=True)
                    -> LSTM(32, tanh, recurrent_dropout=0.2) -> Dense(1)
```

**Multivariate GRU (`model_multi_gru`):**
```
Reshape((7, 4)) -> GRU(32, tanh, recurrent_dropout=0.2, return_sequences=True)
               -> GRU(32, tanh, recurrent_dropout=0.2) -> Dense(1)
```

| Model | MAE | RMSE | MASE |
|-------|-----|------|------|
| Multivariate Dense | 0.960 | 1.471 | 0.709 |
| Multivariate LSTM (Tanh) | 0.942 | 1.462 | 0.695 |
| Multivariate GRU (Tanh) | 0.942 | 1.463 | 0.695 |

The multivariate LSTM and GRU achieve the best MASE across the entire project (0.695), though the improvement over the univariate LSTM (0.696) is negligibly small — a pattern explained in the cross-cutting observations below.

---

### GRU Network

The **Gated Recurrent Unit (GRU)** is introduced in the returns notebook as a leaner alternative to LSTM. GRUs use two gates (reset and update) instead of LSTM's three (input, forget, output), making them computationally lighter while achieving comparable sequential learning. The GRU and LSTM produce essentially identical results here (MASE 0.695 each), consistent with the general finding that on shorter sequences with relatively weak autocorrelation, the architectural difference between the two is negligible. The best-performing multivariate GRU (`model_multi_gru`) is saved as a `.keras` artifact.

---

## Cross-Cutting Observations

### Why Dense and RNN Models Perform Similarly

Across both notebooks, the Dense networks perform almost identically to the LSTM and GRU networks. In the returns notebook, the univariate Dense (MASE 0.709) is only marginally behind the univariate LSTM (0.696), and the multivariate Dense (0.709) trails the multivariate LSTM and GRU (0.695) by a negligible margin.

The reason is rooted in the nature of the signal. Dense networks with a sliding window operate on a fixed-length vector of past observations and learn a weighted combination of those lagged values — they have no explicit mechanism for capturing sequential dependencies across time steps. LSTMs and GRUs, by contrast, maintain a hidden state that can theoretically carry information across arbitrarily long sequences. On a signal with strong temporal dependencies and long-range patterns, this architectural advantage would be decisive.

Daily stock returns, however, are close to white noise. The ADF results and ACF/PACF plots confirm that the autocorrelation of the returns series is weak and decays rapidly to near zero within a few lags. There is simply not enough sequential structure to exploit beyond the immediate recent window. A Dense network reading a 7-day window captures essentially the same information as an LSTM processing the same 7 time steps, because the signal does not contain the kind of long-range dependencies that the LSTM's recurrent memory is designed for. When the signal provides no advantage to recurrence, the architectural complexity of an LSTM does not translate into predictive gains.

This is an important finding: architectural sophistication is not a substitute for a well-specified problem. On a near-random series, simple and complex models converge to the same performance floor.

---

### Why Multivariate Offers Only Marginal Gains

Adding EMA-20, EMA-5, and MACD alongside the raw return series produces only a negligible improvement over the univariate baseline — the best multivariate model (MASE 0.695) barely edges the best univariate model (MASE 0.696).

This result is logically consistent with the nature of the features being added. All three indicators are computed directly from the price or return series itself — they are deterministic transformations of the same underlying signal. EMA-5 is a weighted average of recent returns. EMA-20 is a longer weighted average of the same returns. MACD is a difference between two such averages. None of these indicators introduces genuinely new information that is independent of the return history the model already has in its 7-day sliding window. They are, in effect, redundant encodings of the same data, expressed at different smoothing scales.

In a setting where the underlying series is near-white-noise, the only available information is recent history — and a univariate model with a 7-day window already has direct access to that. The multivariate features encode the same history through a different mathematical lens, but because the signal is not richly structured, this alternate encoding unlocks no additional predictive power. A meaningful multivariate improvement would require genuinely exogenous features — macroeconomic indicators, earnings data, sentiment scores, or other information sources that are independent of price history.

---

### The MAPE Problem in Returns Forecasting

MAPE is a standard evaluation metric in time-series forecasting and is included throughout this project for consistency and comparability. However, it must be interpreted with caution — and in the returns setting, it must not be used for ranking at all.

MAPE divides the absolute error at each observation by the magnitude of the true value. For raw prices, this is well-behaved: NFLX prices range from approximately $1 to $694, so the denominator is always substantial and the percentage errors are bounded. For returns, the true value is a daily dollar change that frequently passes through values close to zero. When the denominator approaches zero, the percentage error for that single observation can reach millions of percent, even if the prediction is only slightly off. The average across thousands of such observations produces a metric value that is numerically enormous but statistically uninformative.

The naive baseline on returns, for instance, produces a MAPE of 414,440% — not because the model is making absurd predictions, but because several denominator values are near zero. The Dense 7-day model produces a MAPE in the tens of thousands of percent for the same structural reason, despite a perfectly sensible MAE of 0.961.

MAPE is nevertheless reported for two reasons. First, it maintains a consistent metric table structure across both notebooks. Second, it serves as a conditional diagnostic: if two models have similar MAE and MASE but very different MAPE values in the returns setting, that indicates one model is making systematically worse predictions specifically on the near-zero return days — a type of conditional failure that MASE would average out. In this project, the MAPE values across all models in the returns setting are all similarly and uniformly inflated, which confirms the issue is structural rather than model-specific. MASE governs all conclusions.

---

## Results Summary

### Raw Prices Notebook

| Model | Window / Config | MAE | MASE | Beats Naive? |
|-------|----------------|-----|------|:---:|
| Naive Forecast | — | 0.939 | 1.000 | — (baseline) |
| ARIMA(1,1,1) | Walk-forward | 0.945 | 1.007 | No |
| Dense (Univariate) | 7-day | 13.75 | 1.030 | No |
| Dense (Univariate) | 10-day | 14.13 | 1.060 | No |
| Dense (Univariate) | 14-day | 14.29 | 1.070 | No |
| LSTM (ReLU) | 7-day | 40.05 | 3.001 | No |
| LSTM (Tanh) | 7-day | 212.32 | 15.91 | No |
| Multivariate LSTM | 7-day x 4 features | 2.32 | 2.470 | No |
| Multivariate Dense | 7-day x 4 features | 1.11 | 1.177 | No |

No model in the raw prices notebook beats the naive forecast on MASE.

### Returns Notebook

| Model | Window / Config | MAE | MASE | Beats Naive? |
|-------|----------------|-----|------|:---:|
| Naive Forecast | — | 1.357 | 1.000 | — (baseline) |
| Dense (Univariate) | 7-day | 0.961 | 0.709 | Yes |
| Dense (Univariate) | 10-day | 0.964 | 0.712 | Yes |
| Dense (Univariate) | 14-day | 0.983 | 0.724 | Yes |
| LSTM (Tanh) | 7-day | 0.943 | 0.696 | Yes |
| LSTM (ReLU) | 7-day | 0.944 | 0.696 | Yes |
| Multivariate Dense | 7-day x 4 features | 0.960 | 0.709 | Yes |
| Multivariate LSTM (Tanh) | 7-day x 4 features | 0.942 | 0.695 | Yes |
| Multivariate GRU (Tanh) | 7-day x 4 features | 0.942 | 0.695 | Yes |

Every model in the returns notebook beats the naive forecast. The best models (Multivariate LSTM and GRU) achieve a MASE of 0.695 — approximately 30% better than the naive baseline.

---

## Key Finding: Random Walk Theory and EMH

The failure of every model in the raw prices notebook is not a bug — it is the expected result according to two foundational theories in financial economics.

### Random Walk Theory

The Random Walk Hypothesis (Malkiel, 1973) states that stock price changes are serially independent — each price movement is statistically unrelated to past movements. Formally:

```
P(t) = P(t-1) + e(t),    where e(t) is white noise
```

If this holds, then the best prediction of tomorrow's price is simply today's price — exactly what the naive model does. The ADF test results in this project directly corroborate this: after first differencing, the series resembles white noise with no significant autocorrelation at any lag. There is no exploitable pattern.

### Efficient Market Hypothesis (EMH)

The EMH (Fama, 1970) asserts that asset prices at any time fully reflect all available information. In its weak form, it claims that no trading strategy based solely on historical price data can consistently generate excess returns — because all such information is already priced in.

Every model in this project uses only historical price and price-derived features (EMA-5, EMA-20, MACD). The fact that none outperforms the naive forecast in the raw prices notebook is direct empirical evidence in support of the weak-form EMH for NFLX over this period.

The raw prices notebook does not fail to predict stock prices. It succeeds in demonstrating that raw stock prices cannot be predicted from their own history, which is precisely what theory predicts.

---

## Why Raw Prices Fail — and What the Returns Notebook Fixes

### The Root Cause: Non-Stationarity

Raw closing prices are non-stationary. They have a time-varying mean (the long-term upward trend of NFLX from approximately $1 to $700) and a time-varying variance (volatility clusters around major events). Models trained on non-stationary data do not learn generalisable patterns — they learn the local trend of the training set and extrapolate it, which fails badly when the test set occupies a different part of the price distribution.

This is why models in the raw prices notebook show low training loss but drastically higher test loss: the distributional shift between the pre-2021 training data and the post-2021 test data (which includes NFLX's peak at approximately $700 and its 2022 crash) is enormous.

### The Corrections Applied in the Returns Notebook

Three simultaneous corrections are applied:

**1. Stationary target.** Raw prices are replaced with absolute returns — the first-differenced series confirmed stationary by the ADF test. The input distribution is now stable across the full training horizon; the model no longer has to extrapolate across a price range spanning two orders of magnitude.

**2. Leakage-free indicator computation.** In the raw prices notebook, EMA and MACD are computed on the full price series before splitting, embedding subtle forward-looking information into the training indicators. In the returns notebook, indicators are computed separately on the training and test dataframes — eliminating this lookahead bias.

**3. No input scaling for multivariate models.** Because the return series and its derived indicators are already near-zero-centred, `StandardScaler` is omitted from the multivariate pipeline entirely. This removes one source of complexity and one potential leakage point.

---

## Overall Impact

This project demonstrates several durable lessons that extend beyond Netflix and beyond stock forecasting.

**Target choice dominates model choice.** The single most impactful decision in this study was switching from raw prices to absolute returns. Every model in the raw prices notebook failed; every model in the returns notebook succeeded. No architectural change — adding LSTM layers, adding features, tuning window sizes — could overcome the fundamental problem of a non-stationary target. Getting the target right is more important than getting the architecture right.

**Architectural complexity does not compensate for a poor signal.** When the underlying series is near-white-noise, simple Dense networks and sophisticated LSTM/GRU networks converge to the same predictive performance. Complexity only pays off when the signal contains the kind of structure — long-range dependencies, non-linear temporal patterns — that the complex architecture is designed to capture.

**More features are not always better.** Adding EMA-20, EMA-5, and MACD produced only a negligible improvement over the univariate models because all three features are deterministic transformations of the same return series. Feature engineering only adds value when the new features carry genuinely independent information.

**The tanh failure in the raw prices notebook is a diagnostic, not a verdict.** Tanh is the theoretically correct activation for LSTM gates. Its catastrophic failure on raw prices (MASE 15.91) is caused entirely by the non-stationarity of the target — the bounded activation saturates when tracking an unbounded trend. The returns notebook confirms this: with a stationary, near-zero-centred target, the tanh LSTM performs well (MASE 0.696), matching the ReLU variant and validating tanh's theoretical correctness for recurrent architectures.

**Empirically validating a negative result is productive.** The raw prices notebook is not wasted work. It provides a rigorous, empirical demonstration of the Random Walk Theory and the weak-form EMH for NFLX over a 23-year period. This grounding makes the improvements in the returns notebook interpretable and credible — they arise from correcting a known, theoretically motivated problem, not from ad hoc tuning.

**Metric selection must match the target distribution.** MAPE, a standard and widely trusted metric, becomes completely uninformative when the target series passes through near-zero values. Blindly relying on it in the returns setting would produce misleading conclusions. MASE, which normalises against the naive forecast, is robust across both target types and should be the default metric for financial forecasting regardless of whether the target is prices or returns.

---

## Model Persistence

Trained models and scalers are saved for downstream inference and deployment.

**Raw prices notebook artifacts (saved via `joblib` and Keras):**
- `Models/scaler_prices.pkl` — univariate `StandardScaler`
- `Models/scaler_multi_input_prices.pkl` — multivariate input scaler
- `Models/scaler_multi_target_prices.pkl` — multivariate target scaler
- `Models/dense_model_uni_prices.keras` — best univariate dense model (7-day window)

**Returns notebook artifacts (saved via Keras):**
- `Models/lstm_uni_returns.keras` — best univariate LSTM on returns (`model_uni_lstm_1`, Tanh)
- `Models/gru_multi_returns.keras` — best multivariate GRU on returns

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `yfinance` | latest | Historical OHLCV data from Yahoo Finance |
| `pandas` | >=1.5 | Data manipulation and feature engineering |
| `numpy` | >=1.23 | Numerical operations and array manipulation |
| `statsmodels` | >=0.13 | ADF test, ACF/PACF plots, ARIMA model |
| `scikit-learn` | >=1.1 | `StandardScaler` for feature normalisation (raw prices notebook) |
| `tensorflow` / `keras` | >=2.12 | Dense, LSTM, and GRU neural network models |
| `matplotlib` | >=3.6 | Visualisation of price series, returns, and predictions |
| `joblib` | latest | Model and scaler persistence |

---

## Project Structure

```
netflix-stock-prediction/
|
|-- netflix_raw_prices.ipynb   # Raw price prediction; empirically validates EMH
|-- netflix_returns.ipynb      # Absolute returns prediction; stationary target
|-- README.md                  # This file
|
+-- Models/
    |-- scaler_prices.pkl                  # Univariate scaler (raw prices)
    |-- scaler_multi_input_prices.pkl      # Multivariate input scaler (raw prices)
    |-- scaler_multi_target_prices.pkl     # Multivariate target scaler (raw prices)
    |-- dense_model_uni_prices.keras       # Best univariate dense model (7-day window)
    |-- lstm_uni_returns.keras             # Best univariate LSTM on returns (Tanh, 7-day)
    +-- gru_multi_returns.keras            # Best multivariate GRU on returns
```

---

## References

- Fama, E. F. (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work.* Journal of Finance, 25(2), 383-417.
- Malkiel, B. G. (1973). *A Random Walk Down Wall Street.* W. W. Norton and Company.
- Box, G. E. P., Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control.* Holden-Day.
- Hochreiter, S., and Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735-1780.
- Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.* arXiv:1406.1078.

---

*This project is for educational and research purposes only. Nothing in this repository constitutes financial advice.*
