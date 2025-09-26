# fx_data.py
import pandas as pd
import os

def get_usdcny_last_week(path: str = "usd_cny_sample.csv") -> pd.DataFrame:
    """
    Load USD/CNY exchange rate data.
    The CSV file must contain at least: ['Date', <price_column>]
    The function returns a DataFrame indexed by date, resampled to daily frequency with forward fill.
    """
    try:
        # Automatically use fallback remote file if local file is not found
        if not os.path.exists(path):
            print("⚠️ Local file not found. Attempting to load from remote URL...")
            path = "https://raw.githubusercontent.com/wanningsun50/fx_risk_monitor_web/main/data/usd_cny_sample.csv"

        df = pd.read_csv(path, parse_dates=['Date'])
        df = df.sort_values('Date').set_index('Date')

        # Resample to daily frequency and forward fill missing values
        df = df.asfreq('D').ffill()

        print("📋 Data preview:")
        print(df.head())
        print("📋 Column names:", list(df.columns))
        return df

    except Exception as e:
        print("❌ Failed to load FX data:", e)
        return pd.DataFrame()
