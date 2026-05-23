# ES Futures — ML Price Direction Predictor

A machine learning project that predicts whether S&P 500 E-mini Futures (ES) will go **UP or DOWN** the next trading day, using historical OHLCV data and a Random Forest classifier.

---

## Project Overview

This project was built as a first step into quantitative finance and ML for trading. It covers the full pipeline from raw data to a working prediction model:

- Loading real futures data via Yahoo Finance
- Feature engineering (returns, moving averages, volatility)
- Statistical analysis of market behavior
- Training and evaluating a Random Forest classifier
- Visualizing results

---

## Results

| Metric | Value |
|---|---|
| Model Accuracy | 53.19% |
| Baseline (random) | 50.00% |
| Training period | 1 year of daily ES data |
| Features used | 4 |

**Feature Importance:**

| Feature | Importance |
|---|---|
| Previous day return | 25.55% |
| 5-day moving average | 23.87% |
| 20-day moving average | 25.54% |
| 5-day volatility | 25.04% |

---

## Key Statistical Findings

| Statistic | Value |
|---|---|
| Mean daily return | 0.10% |
| Daily volatility (std) | 0.76% |
| Skewness | -0.12 (slight negative) |
| Kurtosis | 1.51 (fat tails) |
| Best single day | +2.86% |
| Worst single day | -2.71% |

**Interpretation:**
- ES futures had ~25% annualized returns over the period
- The return distribution shows fat tails — extreme moves happen more than a normal distribution predicts
- Negative skew confirms markets fall faster than they rise
- Volatility clustering is visible — high vol periods cluster together

---

## Project Structure

```
es-ml-project/
│
├── analysis.py          # Main script — data loading, stats, ML model, charts
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Setup & Usage

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/es-ml-project.git
cd es-ml-project
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the analysis**
```bash
python3 analysis.py
```

This will:
- Download 1 year of ES futures data
- Print descriptive statistics
- Train and evaluate the ML model
- Save `es_analysis.png` with 4 charts

---

## How It Works

### 1. Data
Historical daily OHLCV data for ES=F (S&P 500 E-mini Futures) downloaded via `yfinance`.

### 2. Features
| Feature | Formula | What it captures |
|---|---|---|
| `prev_return` | yesterday's % change | Short-term momentum |
| `mavg_5` | 5-day rolling mean | Weekly trend |
| `mavg_20` | 20-day rolling mean | Monthly trend |
| `volatility` | 5-day rolling std dev | Market uncertainty |

### 3. Target
Binary classification:
- `1` = next day closes higher (UP)
- `0` = next day closes lower (DOWN)

### 4. Model
Random Forest Classifier with 100 trees, trained on 80% of data and tested on the most recent 20%.

---

## What's Next

Planned improvements to push accuracy above 55%:

- [ ] Add RSI (Relative Strength Index)
- [ ] Add MACD indicator
- [ ] Add order book imbalance features
- [ ] Try LSTM neural network
- [ ] Add options market data (implied volatility, put/call ratio)
- [ ] Implement walk-forward validation

---

## Learning Resources

Papers that informed this project:
- Fischer & Krauss (2018) — Deep Learning for Stock Prediction
- Cont, Kukanov & Stoikov (2014) — Price Impact of Order Book Events
- Jegadeesh & Titman (1993) — Momentum Strategy

---

## Disclaimer

This project is for **educational purposes only**. Nothing here constitutes financial advice. Past performance does not guarantee future results.

---

*Built while learning ML for quantitative finance from scratch.*
