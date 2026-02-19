# Trader Performance vs Market Sentiment Analysis

## Overview
This project analyzes how Bitcoin market sentiment (Fear vs Greed) influences trader behavior and performance on Hyperliquid. By integrating the Fear & Greed Index with historical trader execution data, we uncover patterns in profitability, risk management, and trading frequency across different market regimes.

## Methodology

### Data Sources
1.  **Bitcoin Fear-Greed Index**: Daily sentiment classification (Extreme Fear, Fear, Neutral, Greed, Extreme Greed).
2.  **Hyperliquid Historical Data**: Trader execution logs including PnL, size, side, and timestamp.

### Data Processing
-   **Alignment**: Both datasets were aligned at a daily granularity.
-   **Metric Engineering**:
    -   `Daily PnL`: Sum of closed PnL per account per day.
    -   `Win Rate`: Percentage of profitable trades.
    -   `Avg Trade Size`: Mean USD value of trades.
    -   `Trade Frequency`: Number of trades per day.
    -   `Long/Short Ratio`: Proportion of Buy orders.

## Key Insights

1.  **Performance vs Sentiment**:
    -   Traders generally experience **reduced profitability** and **higher volatility** during "Fear" regimes.
    -   "Greed" regimes often correlate with higher win rates but also higher risk-taking.

2.  **Behavioral Changes**:
    -   **Over-trading in Greed**: Traders tend to increase both trade size and frequency when sentiment is high.
    -   **Defensive in Fear**: Trading activity drops during Fear, with some traders reducing exposure.

3.  **Trader Segmentation**:
    -   **High-Volume Traders**: Show more consistent performance across regimes but are not immune to Fear-induced drawdowns.
    -   **Retail/Small Traders**: Highly sensitive to sentiment, often entering late in Greed phases.

## Actionable Strategies

Based on the analysis, we recommend the following strategies:

### Strategy 1: Regime-Based Risk Management
-   **Context**: Fear regimes bring unpredictability and sharp drawdowns.
-   **Action**: Automatically **reduce leverage and position size by 25-50%** when the Fear & Greed Index drops below 40 (Fear).
-   **Goal**: Preserve capital during high-volatility/downside periods.

### Strategy 2: Counter-Cyclical Discipline
-   **Context**: Greed regimes (Index > 60) encourage overconfidence and over-trading.
-   **Action**: Cap **daily trade frequency** and **max total exposure** during Extreme Greed to prevent giving back gains.
-   **Goal**: Avoid "euphoria" losses where discipline breaks down.

## Predictive Modeling
A Random Forest model was trained to predict next-day profitability.
-   **Key Features**: Sentiment Value, Trade Frequency, Avg Trade Size.
-   **Result**: Sentiment is a significant predictor of aggregate trader success.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Analysis**:
    To regenerate the processed data and view the analysis logic:
    ```bash
    python generate_data.py
    ```
    Or open `analysis.ipynb` in Jupyter.

3.  **Launch Dashboard**:
    ```bash
    streamlit run dashboard.py
    ```
