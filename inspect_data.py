import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

try:
    fg = pd.read_csv('data/fear_greed_index.csv')
    print("Fear Greed Columns:")
    for col in fg.columns:
        print("- " + col)
    print("\nFear Greed Head:")
    print(fg.head().to_string())
except Exception as e:
    print("Error reading fear_greed_index.csv:", e)

print("\n" + "="*20 + "\n")

try:
    hd = pd.read_csv('data/historical_data.csv', nrows=5)
    print("Historical Data Columns:")
    for col in hd.columns:
        print("- " + col)
    print("\nHistorical Data Head:")
    print(hd.head().to_string())
except Exception as e:
    print("Error reading historical_data.csv:", e)
