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
ax1.set_x_
