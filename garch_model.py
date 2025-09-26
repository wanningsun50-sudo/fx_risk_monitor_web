import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.stattools import adfuller

# ========= 自动模型选择（带保护） =========
def _select_best_model(y_pct: pd.Series):
    from arch import arch_model
    import numpy as np

    best_aic = np.inf
    best_bic = np.inf
    best_res = None
    best_spec = ""

    vol_fams = ['GARCH', 'EGARCH', 'GJR-GARCH']
    dists = ['normal', 't', 'skewt', 'ged']

    for vol in vol_fams:
        for dist in dists:
            for p in range(1, 3):
                for q in range(1, 3):
                    for lags in range(0, 3):
                        try:
                            mean_model = 'Zero' if lags == 0 else 'AR'
                            o = 1 if vol != 'GARCH' else 0
                            am = arch_model(
                                y_pct,
                                mean=mean_model,
                                lags=lags,
                                vol=vol,
                                p=p,
                                o=o,
                                q=q,
                                dist=dist,
                                rescale=True
                            )
                            res = am.fit(disp='off', show_warning=False)

                            # 🛡️ 自由度检查
                            if dist in ['t', 'skewt']:
                                nu = float(res.params.get("nu", 8.0))
                                if nu < 2.5 or nu > 100:
                                    continue

                            # 🛡️ 波动率合理性检查
                            vol_series = res.conditional_volatility
                            if (vol_series < 1e-6).any() or (vol_series > 1e3).any():
                                continue

                            # ✅ NEW: 标准化残差限制
                            resid = res.resid
                            std_resid = resid / (vol_series + 1e-8)
                            if np.abs(std_resid).max() > 100:
                                continue

                            aic, bic = float(res.aic), float(res.bic)

                            is_better = False
                            if aic < best_aic - 1e-8:
                                is_better = True
                            elif abs(aic - best_aic) <= 1e-8 and bic < best_bic:
                                is_better = True

                            if is_better:
                                best_aic, best_bic = aic, bic
                                best_res = res
                                best_spec = f"mean={mean_model}({lags}) + {vol}({p},{q},o={o})，dist={dist}"

                        except Exception:
                            continue

    if best_res is None:
        raise RuntimeError("自动模型选择失败，请检查数据或缩小搜索空间。")

    return best_res, (best_aic, best_bic, best_spec)

# ========= 波动率计算（自动+阈值） =========
def compute_volatility(df: pd.DataFrame):
    col = df.columns[0]
    out = df[[col]].copy()

    out['log_return'] = 100.0 * np.log(out[col] / out[col].shift(1))
    out = out.dropna()

    try:
        adf_p = adfuller(out['log_return'])[1]
        print(f"ADF p-value on returns = {adf_p:.4f}（<0.05 常见为平稳）")
    except Exception:
        adf_p = None

    y_pct = out['log_return'].astype(float)

    best_res, (aic, bic, spec) = _select_best_model(y_pct)
    print(f"最佳模型: {spec} | AIC={aic:.2f}, BIC={bic:.2f}")

    out['volatility_raw'] = best_res.conditional_volatility.astype(float)
    out['volatility'] = out['volatility_raw'].rolling(window=3, min_periods=1).mean()

    n = len(out)
    lookback = max(20, min(120, int(n * 0.8)))
    roll_q95 = out['volatility'].rolling(lookback, min_periods=max(10, lookback // 2)).quantile(0.95)

    threshold = float(roll_q95.dropna().iloc[-1]) if roll_q95.notna().any() else float(out['volatility'].quantile(0.95))
    latest_vol = float(out['volatility'].iloc[-1])
    warning = bool(latest_vol > threshold)

    return out, warning, latest_vol, threshold


# ========= 未来价格预测（逐日滚动 + 自动选模型） =========
def forecast_future_prices_rolling(df: pd.DataFrame, steps: int = 5, alpha: float = 0.05, dist_for_ci: str = "t"):
    from statistics import NormalDist

    try:
        from scipy.stats import t as t_dist, norm
        has_scipy = True
    except Exception:
        has_scipy = False
        norm = NormalDist()
        t_dist = None

    def _z(df_=None):
        if dist_for_ci == "t" and has_scipy and df_ is not None:
            return float(t_dist.ppf(1 - alpha / 2, df=df_))
        return float(norm.ppf(1 - alpha / 2))

    col = df.columns[0]
    prices_history = df[col].astype(float).copy()
    last_date = df.index[-1]

    preds = []
    uppers = []
    lowers = []

    for i in range(steps):
        log_return = 100.0 * np.log(prices_history / prices_history.shift(1)).dropna()

        try:
            res, (aic, bic, spec) = _select_best_model(log_return)
            print(f"✅ 第 {i+1} 天选择模型: {spec}")
        except Exception as e:
            print(f"❌ 第 {i+1} 天模型选择失败：{e}")
            break

        forecast = res.forecast(horizon=1, reindex=False)
        mu = forecast.mean.iloc[-1, 0]
        sigma = np.sqrt(forecast.variance.iloc[-1, 0])
        sigma = min(sigma, 20.0)

        dfree = float(res.params.get("nu", 8.0)) if (dist_for_ci == "t" and has_scipy) else None
        z = _z(dfree)

        mu = mu / 100.0
        sigma = sigma / 100.0

        prev_price = prices_history.iloc[-1]
        pred_price = prev_price * np.exp(mu)
        upper_price = prev_price * np.exp(mu + z * sigma)
        lower_price = prev_price * np.exp(mu - z * sigma)

        if not (5.5 <= pred_price <= 8.5):
            print(f"⚠️ 第 {i+1} 天预测值异常：{pred_price:.4f}，终止后续预测")
            break

        preds.append(pred_price)
        uppers.append(upper_price)
        lowers.append(lower_price)

        next_date = last_date + pd.Timedelta(days=1)
        prices_history.loc[next_date] = pred_price
        last_date = next_date

    return np.array(preds), np.array(uppers), np.array(lowers)
