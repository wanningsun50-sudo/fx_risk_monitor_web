# main/app.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import warnings
import matplotlib

warnings.filterwarnings("ignore", category=RuntimeWarning)

from fx_data import get_usdcny_last_week
from garch_model import compute_volatility, forecast_future_prices_rolling

# ✅ 使用兼容 Streamlit Cloud 的中文字体（移除 SimHei）
try:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Microsoft YaHei', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    st.warning(f"⚠️ 字体设置失败：{e}")

# ✅ Streamlit 页面开始
st.set_page_config(page_title="外汇风险监测", layout="wide")
st.title("📈 USD/CNY 外汇风险监测系统")

# 加载数据
df = get_usdcny_last_week()
if df.empty:
    st.error("❌ 无法获取汇率数据，终止分析。")
    st.stop()

st.info("✅ 汇率数据加载成功，开始计算波动率与风险预警...")

# 计算波动率
df_result, warning, latest_vol, threshold = compute_volatility(df)

# 展示当前风险状态
st.subheader("📊 当前波动率分析")
st.write(f"**最新波动率（%）:** `{latest_vol:.4f}`")
st.write(f"**滚动95%分位阈值（%）:** `{threshold:.4f}`")
if warning:
    st.error("🚨 当前波动率高于阈值，触发风险预警！")
else:
    st.success("✅ 当前波动率正常，无风险预警。")

# === 图1：汇率走势 ===
col_name = df_result.columns[0]
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df_result.index, df_result[col_name], label=col_name, color='steelblue')
ax1.set_title(f"{col_name} 汇率走势")
ax1.set_xlabel("日期")
ax1.set_ylabel("汇率")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# === 图2：波动率趋势 ===
vol_pct = df_result['volatility']
n = len(vol_pct)
win = max(20, min(60, n // 2))
minp = max(10, win // 2)
q95 = vol_pct.rolling(window=win, min_periods=minp).quantile(0.95)
if q95.isna().all():
    q95 = vol_pct.expanding(min_periods=10).quantile(0.95)

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(df_result.index, vol_pct, label='条件波动率 (%)', color='orange')
ax2.plot(q95.index, q95.values, '--', label='滚动95%分位', color='red')
ax2.set_title("USD/CNY 波动率趋势（单位：%）")
ax2.set_xlabel("日期")
ax2.set_ylabel("波动率 (%)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# === 图3：未来预测图 ===
st.subheader("🔮 未来汇率预测")
for steps in [5, 15]:
    st.markdown(f"### 📈 未来 {steps} 天汇率预测")

    prices, upper, lower = forecast_future_prices_rolling(df, steps=steps, alpha=0.05, dist_for_ci="t")

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                 periods=prices.shape[0], freq='D')

    mask = np.isfinite(prices) & np.isfinite(upper) & np.isfinite(lower)
    prices, upper, lower, future_dates = prices[mask], upper[mask], lower[mask], future_dates[mask]

    if prices.size == 0:
        st.warning(f"⚠️ 预测结果为空，跳过未来 {steps} 天预测。")
        continue

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(future_dates, prices, label='预测中枢', color='blue')
    ax3.fill_between(future_dates, lower, upper, alpha=0.1, label='置信区间', color='skyblue')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.4f}"))
    for x, y in zip(future_dates, prices):
        ax3.text(x, y, f"{y:.4f}", fontsize=8, ha='center', va='bottom', color='blue')
    ax3.set_title(f"未来 {steps} 天 USD/CNY 逐日滚动预测")
    ax3.set_xlabel("日期")
    ax3.set_ylabel("汇率")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)


