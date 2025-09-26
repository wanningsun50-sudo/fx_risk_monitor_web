# main.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from fx_data import get_usdcny_last_week
from garch_model import compute_volatility, forecast_future_prices_rolling

# 中文显示
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']  # ✅ 安全替代 SimHei
matplotlib.rcParams['axes.unicode_minus'] = False         # 解决负号乱码

def main():
    print("📈 正在获取汇率数据...")
    df = get_usdcny_last_week()
    if df.empty:
        print("❌ 数据为空，终止分析。")
        return

    print("⚙️ 正在计算波动率与风险...")
    df_result, warning, latest_vol, threshold = compute_volatility(df)

    print(f"\n📊 最新波动率（%）：{latest_vol:.4f}")
    print(f"📉 预警阈值（滚动95%分位，%）：{threshold:.4f}")
    print("🚨 风险预警：当前波动率高" if warning else "✅ 波动率正常")

    col_name = df_result.columns[0]

    # 1) 汇率走势
    plt.figure(figsize=(12, 4))
    plt.plot(df_result.index, df_result[col_name], color='steelblue', label=col_name)
    plt.title(f"{col_name} 汇率走势")
    plt.xlabel("日期"); plt.ylabel("汇率"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # 2) 波动率趋势（单位：%）
    vol_pct = df_result['volatility']
    n = len(vol_pct)
    win = max(20, min(60, n // 2))
    minp = max(10, win // 2)
    q95 = vol_pct.rolling(window=win, min_periods=minp).quantile(0.95)

    if q95.isna().all():
        q95 = vol_pct.expanding(min_periods=10).quantile(0.95)

    plt.figure(figsize=(12, 4))
    plt.plot(df_result.index, vol_pct, label='条件波动率 (%)', color='orange', linewidth=2)
    plt.plot(q95.index, q95.values, linestyle='--', color='red', linewidth=2, label='滚动95%分位')
    plt.title(f"USD/CNY 波动率趋势（单位：%）｜当前波动率: {latest_vol:.2f}%｜阈值: {threshold:.2f}%")
    plt.xlabel("日期")
    plt.ylabel("波动率 (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3) 逐日滚动预测：未来 5 天 与 15 天
    for steps in [5, 15]:
        print(f"\n📈 正在逐日滚动预测未来 {steps} 天汇率...")
        prices, upper, lower = forecast_future_prices_rolling(
            df, steps=steps, alpha=0.05, dist_for_ci="t"
        )

        # 用返回长度对齐日期
        future_dates = pd.date_range(
            start=df.index[-1] + pd.Timedelta(days=1),
            periods=prices.shape[0], freq='D'
        )

        # 过滤非有限值，三者同步
        mask = np.isfinite(prices) & np.isfinite(upper) & np.isfinite(lower)
        future_dates = future_dates[mask]
        prices = prices[mask]
        upper = upper[mask]
        lower = lower[mask]

        if prices.size == 0:
            print("⚠️ 预测序列为空（或全是非有限值），本次绘图跳过。")
            continue

        # ✅ 在原代码内
        plt.figure(figsize=(10, 5))
        plt.plot(future_dates, prices, label='预测中枢', color='blue')
        plt.fill_between(future_dates, lower, upper, alpha=0.1, label='置信区间', color='skyblue')

        # ✅ 显示更多小数位
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.4f}"))

        # ✅ 在每个点上显示数值
        for x, y in zip(future_dates, prices):
            plt.text(x, y, f"{y:.4f}", fontsize=8, ha='center', va='bottom', color='blue')

        plt.title(f'未来 {steps} 天 USD/CNY 逐日滚动预测（对数收益空间构造区间）')
        plt.xlabel('日期')
        plt.ylabel('汇率')
        plt.legend()
        plt.grid(True)
        plt.tight_layout(pad=2)
        plt.show()


if __name__ == "__main__":
    main()

