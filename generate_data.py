import pandas as pd
import numpy as np

# Settings
pd.set_option('display.max_columns', None)

print("Loading data...")
# Load Fear and Greed Index
fg = pd.read_csv('data/fear_greed_index.csv')
fg['timestamp'] = pd.to_datetime(fg['timestamp'], unit='s')
fg['date'] = fg['timestamp'].dt.date
fg = fg.sort_values('date')

# Load Historical Execution Data
hd = pd.read_csv('data/historical_data.csv')
hd['Timestamp'] = pd.to_datetime(hd['Timestamp'], unit='ms')
hd['date'] = hd['Timestamp'].dt.date

# Check if 'Crossed' is boolean or string
# If string 'true'/'false', convert. If boolean, use as is.
if hd['Crossed'].dtype == object:
    hd['Crossed'] = hd['Crossed'].astype(str).str.lower() == 'true'

# Calculate Daily Trader Metrics
print("Calculating metrics...")
daily_stats = hd.groupby(['Account', 'date']).apply(
    lambda x: pd.Series({
        'daily_pnl': x['Closed PnL'].sum(),
        'total_volume': x['Size USD'].sum(),
        'trade_count': len(x),
        'win_rate': (x['Closed PnL'] > 0).mean(),
        'avg_trade_size': x['Size USD'].mean(),
        'long_ratio': (x['Side'] == 'BUY').mean(),
        'fee_sum': x['Fee'].sum(),
        'fee_avg': x['Fee'].mean(),
        'crossed_ratio': x['Crossed'].mean(),
    })
).reset_index()

# Coin Volume Stats (Global, not per account/day for the dashboard sum, but we might want per day for deeper analysis)
# For the specific chart "Volume by Coin", we can pre-calculate it or do it in dashboard.
# Let's keep coin stats separate or just compute in dashboard from raw if needed?
# The React dashboard has a specific "Volume by Coin" chart.
# Let's save a separate coin stats file? Or just use the daily stats.
# Actually, daily stats lose the "Coin" detail.
# Let's add top coin per day? No, that's messy.
# We will load raw data in dashboard for "Coin" view? 
# Or better: Create a secondary dataset for Coin stats.
print("Calculating coin stats...")
coin_stats = hd.groupby('Coin').agg({
    'Size USD': 'sum',
    'Closed PnL': 'sum'
}).reset_index().rename(columns={'Size USD': 'vol', 'Closed PnL': 'pnl'})
coin_stats.to_csv('coin_stats.csv', index=False)


# Merge with Fear and Greed Index
print("Merging data...")
merged_df = pd.merge(daily_stats, fg[['date', 'value', 'classification']], on='date', how='inner')
merged_df['sentiment_regime'] = merged_df['classification']

# Define Segments (Global quantiles to fix segments across time)
# Note: React code uses fixed segments "High-Size", "Low-Size" etc.
# We will stick to the quantile method but label them similarly to React code if possible.
# React: High-Size, Low-Size, Frequent, Infrequent, Consistent.
# "Consistent" is a performance metric, not just size/freq.
# Let's clean up segments.
merged_df['size_segment'] = pd.qcut(merged_df['avg_trade_size'], 3, labels=['Low-Size', 'Mid-Size', 'High-Size'])
merged_df['freq_segment'] = pd.qcut(merged_df['trade_count'], 3, labels=['Infrequent', 'Normal', 'Frequent'])

# "Consistent" segment logic: Profitable > 50% of days? Or just a label for top performers?
# React code "Consistent" has specific stats. I will just rely on size/freq for now in the main data.

# Save processed data for Dashboard
merged_df.to_csv('processed_data_v2.csv', index=False)
print("Processed data saved to processed_data_v2.csv and coin_stats.csv")
