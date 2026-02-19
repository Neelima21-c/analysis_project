import pandas as pd
import numpy as np

def load_and_process_data():
    print("Loading data...")
    # Load Fear and Greed
    fg = pd.read_csv('data/fear_greed_index.csv')
    fg['timestamp'] = pd.to_datetime(fg['timestamp'], unit='s')
    fg['date'] = fg['timestamp'].dt.date
    fg = fg.sort_values('date')
    print(f"Fear Greed loaded: {len(fg)} rows")

    # Load Historical Data
    # Columns: Account, Coin, Execution Price, Size USD, Side, Timestamp, Closed PnL
    hd = pd.read_csv('data/historical_data.csv')
    hd['Timestamp'] = pd.to_datetime(hd['Timestamp'], unit='ms') # Assuming ms based on 1000s in other timestamp
    hd['date'] = hd['Timestamp'].dt.date
    print(f"Historical Data loaded: {len(hd)} rows")

    # Filter for Bitcoin if necessary? The objective mentions "Bitcoin market sentiment" 
    # and "Hyperliquid historical trader execution data". 
    # Does historical data contain only BTC? Let's check unique coins.
    print("Unique Coins:", hd['Coin'].unique())
    
    # Engineer Daily Trader Metrics
    print("Engineering metrics...")
    
    # Group by Account and Date
    daily_stats = hd.groupby(['Account', 'date']).apply(
        lambda x: pd.Series({
            'daily_pnl': x['Closed PnL'].sum(),
            'total_volume': x['Size USD'].sum(),
            'trade_count': len(x),
            'win_rate': (x['Closed PnL'] > 0).mean(),
            'avg_trade_size': x['Size USD'].mean(),
            'long_ratio': (x['Side'] == 'B').mean() # Assuming B is Buy/Long? Need to verify Side values.
        })
    ).reset_index()
    
    print("Daily stats created. Rows:", len(daily_stats))
    print(daily_stats.head())

    return fg, daily_stats

if __name__ == "__main__":
    try:
        fg, daily_stats = load_and_process_data()
        
        # Check Side column values
        hd = pd.read_csv('data/historical_data.csv', usecols=['Side'], nrows=1000)
        print("Unique Sides:", hd['Side'].unique())
        
    except Exception as e:
        print(f"Error: {e}")
