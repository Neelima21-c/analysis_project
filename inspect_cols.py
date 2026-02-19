import pandas as pd

with open('columns.txt', 'w') as f:
    try:
        fg = pd.read_csv('data/fear_greed_index.csv')
        f.write("Fear Greed Columns:\n")
        for col in fg.columns:
            f.write("- " + col + "\n")
    except Exception as e:
        f.write("Error reading fear_greed_index.csv: " + str(e) + "\n")

    f.write("\n")

    try:
        hd = pd.read_csv('data/historical_data.csv', nrows=5)
        f.write("Historical Data Columns:\n")
        for col in hd.columns:
            f.write("- " + col + "\n")
    except Exception as e:
        f.write("Error reading historical_data.csv: " + str(e) + "\n")
