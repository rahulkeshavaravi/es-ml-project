"""
ES Futures - Machine Learning Price Direction Predictor
Author: Rahul
Description: Predicts whether S&P 500 futures will go UP or DOWN the next day
             using machine learning on historical OHLCV data.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading ES Futures data...")
df = yf.download("ES=F", period="1y", interval="1d")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
df["daily_return"] = df["Close"].pct_change()
df["prev_return"]  = df["daily_return"].shift(1)   # Yesterday's return
df["mavg_5"]       = df["Close"].rolling(5).mean()  # 5-day moving average
df["mavg_20"]      = df["Close"].rolling(20).mean() # 20-day moving average
df["volatility"]   = df["daily_return"].rolling(5).std()  # 5-day volatility

# Target: 1 if tomorrow goes UP, 0 if DOWN
df["target"] = (df["daily_return"].shift(-1) > 0).astype(int)
df = df.dropna()

# ─────────────────────────────────────────────
# 3. DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────
returns = df["daily_return"] * 100

print("\n===== DESCRIPTIVE STATISTICS =====")
print(f"Mean daily return:   {returns.mean():.4f}%")
print(f"Median daily return: {returns.median():.4f}%")
print(f"Std deviation:       {returns.std():.4f}%")
print(f"Skewness:            {returns.skew():.4f}")
print(f"Kurtosis:            {returns.kurt():.4f}")
print(f"Best day:            {returns.max():.4f}%")
print(f"Worst day:           {returns.min():.4f}%")
print(f"\n68% of days fall between: {returns.mean() - returns.std():.2f}% and {returns.mean() + returns.std():.2f}%")
print(f"95% of days fall between: {returns.mean() - 2*returns.std():.2f}% and {returns.mean() + 2*returns.std():.2f}%")

# ─────────────────────────────────────────────
# 4. TRAIN ML MODEL
# ─────────────────────────────────────────────
features = ["prev_return", "mavg_5", "mavg_20", "volatility"]
X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy    = accuracy_score(y_test, predictions)

print("\n===== ML MODEL RESULTS =====")
print(f"Model Accuracy: {round(accuracy * 100, 2)}%")
print("\nDetailed Report:")
print(classification_report(y_test, predictions, target_names=["DOWN", "UP"]))

print("Feature Importance:")
for feature, importance in zip(features, model.feature_importances_):
    print(f"  {feature}: {round(importance * 100, 2)}%")

# ─────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("ES Futures - Statistical Analysis", fontsize=14, fontweight="bold")

# Daily returns
axes[0, 0].plot(df.index, returns, color="steelblue", alpha=0.7, linewidth=0.8)
axes[0, 0].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
axes[0, 0].set_title("Daily Returns Over Time")
axes[0, 0].set_ylabel("Return %")

# Distribution
axes[0, 1].hist(returns, bins=40, color="seagreen", alpha=0.75, edgecolor="white")
axes[0, 1].axvline(returns.mean(),   color="red",    linestyle="--", label="Mean",   linewidth=1.5)
axes[0, 1].axvline(returns.median(), color="orange", linestyle="--", label="Median", linewidth=1.5)
axes[0, 1].set_title("Distribution of Returns")
axes[0, 1].legend()

# Cumulative returns
cumulative = (1 + returns / 100).cumprod() - 1
axes[1, 0].plot(df.index, cumulative * 100, color="purple", linewidth=1.2)
axes[1, 0].fill_between(df.index, cumulative * 100, alpha=0.1, color="purple")
axes[1, 0].set_title("Cumulative Return %")
axes[1, 0].set_ylabel("Total Return %")

# Rolling volatility
rolling_vol = returns.rolling(20).std()
axes[1, 1].plot(df.index, rolling_vol, color="crimson", linewidth=1.0)
axes[1, 1].set_title("Rolling 20-day Volatility")
axes[1, 1].set_ylabel("Std Dev %")

plt.tight_layout()
plt.savefig("es_analysis.png", dpi=150, bbox_inches="tight")
print("\nChart saved as es_analysis.png")
