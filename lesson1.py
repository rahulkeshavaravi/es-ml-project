import yfinance as yf
import pandas as pd

# Load ES futures
df = yf.download("ES=F", period="1y", interval="1d")
df["daily_return"] = df["Close"].pct_change() * 100
df = df.dropna()

returns = df["daily_return"]

print("===== DESCRIPTIVE STATISTICS =====")
print(f"Mean daily return:   {returns.mean():.4f}%")
print(f"Median daily return: {returns.median():.4f}%")
print(f"Std deviation:       {returns.std():.4f}%")
print(f"Skewness:            {returns.skew():.4f}")
print(f"Kurtosis:            {returns.kurt():.4f}")
print(f"Best day:            {returns.max():.4f}%")
print(f"Worst day:           {returns.min():.4f}%")

print("\n===== WHAT THIS MEANS =====")
print(f"On average ES moves: {returns.mean():.2f}% per day")
print(f"68% of days fall between: {returns.mean() - returns.std():.2f}% and {returns.mean() + returns.std():.2f}%")
print(f"95% of days fall between: {returns.mean() - 2*returns.std():.2f}% and {returns.mean() + 2*returns.std():.2f}%")

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

df = yf.download("ES=F", period="1y", interval="1d")
df["daily_return"] = df["Close"].pct_change() * 100
df = df.dropna()

returns = df["daily_return"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("ES Futures - Statistical Analysis", fontsize=14)

# 1. Daily returns over time
axes[0,0].plot(df.index, returns, color="blue", alpha=0.7)
axes[0,0].axhline(y=0, color="black", linestyle="--")
axes[0,0].set_title("Daily Returns Over Time")
axes[0,0].set_ylabel("Return %")

# 2. Distribution of returns
axes[0,1].hist(returns, bins=40, color="green", alpha=0.7, edgecolor="black")
axes[0,1].axvline(returns.mean(), color="red", linestyle="--", label="Mean")
axes[0,1].axvline(returns.median(), color="orange", linestyle="--", label="Median")
axes[0,1].set_title("Distribution of Returns")
axes[0,1].legend()

# 3. Cumulative returns
cumulative = (1 + returns/100).cumprod() - 1
axes[1,0].plot(df.index, cumulative * 100, color="purple")
axes[1,0].set_title("Cumulative Return %")
axes[1,0].set_ylabel("Total Return %")

# 4. Rolling volatility
rolling_vol = returns.rolling(20).std()
axes[1,1].plot(df.index, rolling_vol, color="red")
axes[1,1].set_title("Rolling 20-day Volatility")
axes[1,1].set_ylabel("Std Dev %")

plt.tight_layout()
plt.savefig("es_analysis.png")
print("Chart saved as es_analysis.png")