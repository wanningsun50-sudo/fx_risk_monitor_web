# app.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")  # ✅ Use Agg backend for headless servers
import matplotlib.pyplot as plt
import streamlit as st
import warnings

from fx_data import get_usdcny_last_week
from garch_model import compute_volatility, forecast_future_prices_rolling

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ✅ Font config (pure English fonts)
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# ✅ Streamlit page config
st.set_page_config(page_title="FX Risk Monitor", layout="wide")
st.title("📈 USD/CNY FX Risk Monitoring System")

# ✅ Load and validate data
df = get_usdcny_last_week()
if df.empty or df.shape[0] < 30:
    st.error("❌ FX data is empty or too short. Cannot proceed.")
    st.stop()

df.index = pd.to_datetime(df.index)
st.success(f"✅ Data loaded: {df.shape[0]} rows")

# ✅ Compute volatility
df_result, warning, latest_vol, threshold = compute_volatility(df)
df_result.index = pd.to_datetime(df_result.index)

# ✅ Display metrics
st.subheader("📊 Volatility Status")
st.metric("Latest Volatility (%)", f"{latest_vol:.4f}")
st.metric("Rolling 95% Threshold (%)", f"{threshold:.4f}")
if warning:
    st.error("🚨 Risk Warning: Volatility exceeds threshold!")
else:
    st.success("✅ Volatility within normal range.")

# === Chart 1: Exchange Rate
col_name = df_result.columns[0]
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df_result.index, df_result[col_name], label="USD/CNY", color="steelblue")
ax1.set_title("USD/CNY Exchange Rate")
ax1.set_xlabel("Date")
ax1.set_ylabel("Exchange Rate")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# === Chart 2: Volatility Trend
vol_pct = df_result["volatility"]
n = len(vol_pct)
win = max(20, min(60, n // 2))
minp = max(10, win // 2)
q95 = vol_pct.rolling(window=win, min_periods=minp).quantile(0.95)
if q95.isna().all():
    q95 = vol_pct.expanding(min_periods=10).quantile(0.95)

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(df_result.index, vol_pct, label="Volatility (%)", color="orange")
ax2.plot(q95.index, q95.values, "--", label="Rolling 95% Quantile", color="red")
ax2.set_title("USD/CNY Volatility Trend")
ax2.set_xlabel("Date")
ax2.set_ylabel("Volatility (%)")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# === Chart 3: Forecast
st.subheader("🔮 Forecasted Exchange Rate")
for steps in [5, 15]:
    st.markdown(f"### 📈 Next {steps}-Day Forecast")

    try:
        prices, upper, lower = forecast_future_prices_rolling(df, steps=steps)
    except Exception as e:
        st.warning(f"⚠️ Forecast failed: {e}")
        continue

    if not (np.isfinite(prices).any() and np.isfinite(upper).any()):
        st.warning(f"⚠️ Empty forecast results for {steps} days. Skipping.")
        continue

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=len(prices), freq="D")

    mask = np.isfinite(prices) & np.isfinite(upper) & np.isfinite(lower)
    prices, upper, lower, future_dates = prices[mask], upper[mask], lower[mask], future_dates[mask]

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(future_dates, prices, label="Forecast", color="blue")
    ax3.fill_between(future_dates, lower, upper, alpha=0.1, label="Confidence Interval", color="skyblue")
    ax3.set_title(f"USD/CNY Forecast: Next {steps} Days")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Exchange Rate")
    ax3.grid(True)
    ax3.legend()

    # Optional: Label some forecast points
    step = max(1, len(prices) // 10)
    for x, y in zip(future_dates[::step], prices[::step]):
        ax3.text(x, y, f"{y:.4f}", fontsize=8, ha="center", va="bottom", color="blue")

    st.pyplot(fig3)
