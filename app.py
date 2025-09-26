import pandas as pd
import numpy as np
import matplotlib

# ✅ Use a non-interactive backend compatible with Streamlit
matplotlib.use("agg")
import matplotlib.pyplot as plt
import streamlit as st
import warnings

from fx_data import get_usdcny_last_week
from garch_model import compute_volatility, forecast_future_prices_rolling

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ✅ Use only English-compatible fonts
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ✅ Streamlit page config
st.set_page_config(page_title="FX Risk Monitor", layout="wide")
st.title("📈 USD/CNY FX Risk Monitoring System")

# ✅ Load data
df = get_usdcny_last_week()
if df.empty:
    st.error("❌ Failed to fetch FX data. Exiting analysis.")
    st.stop()

df.index = pd.to_datetime(df.index)

st.info("✅ FX data loaded. Calculating volatility and risk warning...")

# ✅ Compute volatility
df_result, warning, latest_vol, threshold = compute_volatility(df)
df_result.index = pd.to_datetime(df_result.index)

# ✅ Show current risk status
st.subheader("📊 Volatility Analysis")
st.write(f"**Latest Volatility (%):** `{latest_vol:.4f}`")
st.write(f"**Rolling 95th Percentile Threshold (%):** `{threshold:.4f}`")
if warning:
    st.error("🚨 Volatility exceeds threshold. Risk warning triggered!")
else:
    st.success("✅ Volatility is within normal range.")

# === Chart 1: Exchange Rate Trend ===
col_name = df_result.columns[0]
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df_result.index, df_result[col_name], label=col_name, color='steelblue')
ax1.set_title(f"{col_name} Exchange Rate Trend", fontsize=12)
ax1.set_xlabel("Date", fontsize=10)
ax1.set_ylabel("Exchange Rate", fontsize=10)
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# === Chart 2: Volatility Trend ===
vol_pct = df_result['volatility']
n = len(vol_pct)
win = max(20, min(60, n // 2))
minp = max(10, win // 2)
q95 = vol_pct.rolling(window=win, min_periods=minp).quantile(0.95)
if q95.isna().all():
    q95 = vol_pct.expanding(min_periods=10).quantile(0.95)

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(df_result.index, vol_pct, label='Conditional Volatility (%)', color='orange')
ax2.plot(q95.index, q95.values, '--', label='Rolling 95th Percentile', color='red')
ax2.set_title("USD/CNY Volatility Trend", fontsize=12)
ax2.set_xlabel("Date", fontsize=10)
ax2.set_ylabel("Volatility (%)", fontsize=10)
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# === Chart 3: Future Forecast ===
st.subheader("🔮 Future Exchange Rate Forecast")
for steps in [5, 15]:
    st.markdown(f"### 📈 {steps}-Day Forecast")

    prices, upper, lower = forecast_future_prices_rolling(df, steps=steps, alpha=0.05, dist_for_ci="t")

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                 periods=prices.shape[0], freq='D')

    mask = np.isfinite(prices) & np.isfinite(upper) & np.isfinite(lower)
    prices, upper, lower, future_dates = prices[mask], upper[mask], lower[mask], future_dates[mask]

    if prices.size == 0:
        st.warning(f"⚠️ Forecast data is empty. Skipping {steps}-day forecast.")
        continue

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(future_dates, prices, label='Forecast Median', color='blue')
    ax3.fill_between(future_dates, lower, upper, alpha=0.1, label='Confidence Interval', color='skyblue')

    max_labels = 10
    step = max(1, len(prices) // max_labels)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.4f}"))
    for x, y in zip(future_dates[::step], prices[::step]):
        ax3.text(x, y, f"{y:.4f}", fontsize=8, ha='center', va='bottom', color='blue')

    ax3.set_title(f"{steps}-Day Rolling Forecast (USD/CNY)", fontsize=12)
    ax3.set_xlabel("Date", fontsize=10)
    ax3.set_ylabel("Exchange Rate", fontsize=10)
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)
