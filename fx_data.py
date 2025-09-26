# fx_data.py
import pandas as pd

def get_usdcny_last_week(path: str = "usd_cny_sample.csv") -> pd.DataFrame:
    """
    读取 USD/CNY 数据，要求至少包含列: ['Date', <price_col>]
    返回按日期索引的 DataFrame（升频到日频并前向填充）
    """
    try:
        df = pd.read_csv(path, parse_dates=['Date'])
        df = df.sort_values('Date').set_index('Date')
        # 统一为日频，缺失用前值填补（外汇休市/节假日）
        df = df.asfreq('D').ffill()
        print("📋 数据预览：")
        print(df.head())
        print("📋 列名：", list(df.columns))
        return df
    except Exception as e:
        print("❌ 获取外汇数据失败：", e)
        return pd.DataFrame()
