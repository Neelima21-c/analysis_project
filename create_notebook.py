import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Trader Performance vs Market Sentiment Analysis\n",
    "\n",
    "## Objective\n",
    "Analyze how Bitcoin market sentiment (Fear vs Greed) influences trader behavior and performance on Hyperliquid.\n",
    "\n",
    "## Data Sources\n",
    "- Bitcoin Fear-Greed Index\n",
    "- Hyperliquid Historical Trader Execution Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Settings\n",
    "pd.set_option('display.max_columns', None)\n",
    "sns.set_style('darkgrid')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading and Preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Fear and Greed Index\n",
    "fg = pd.read_csv('data/fear_greed_index.csv')\n",
    "fg['timestamp'] = pd.to_datetime(fg['timestamp'], unit='s')\n",
    "fg['date'] = fg['timestamp'].dt.date\n",
    "fg = fg.sort_values('date')\n",
    "print(f\"Fear Greed Data: {fg.shape}\")\n",
    "print(fg.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Historical Execution Data\n",
    "hd = pd.read_csv('data/historical_data.csv')\n",
    "hd['Timestamp'] = pd.to_datetime(hd['Timestamp'], unit='ms')\n",
    "hd['date'] = hd['Timestamp'].dt.date\n",
    "print(f\"Historical Data: {hd.shape}\")\n",
    "print(hd.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Metric Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate Daily Trader Metrics\n",
    "\n",
    "def calculate_daily_metrics(x):\n",
    "    return pd.Series({\n",
    "        'daily_pnl': x['Closed PnL'].sum(),\n",
    "        'total_volume': x['Size USD'].sum(),\n",
    "        'trade_count': len(x),\n",
    "        'win_rate': (x['Closed PnL'] > 0).mean(),\n",
    "        'avg_trade_size': x['Size USD'].mean(),\n",
    "        'long_ratio': (x['Side'] == 'BUY').mean()\n",
    "    })\n",
    "\n",
    "daily_stats = hd.groupby(['Account', 'date']).apply(calculate_daily_metrics).reset_index()\n",
    "print(f\"Daily Stats: {daily_stats.shape}\")\n",
    "print(daily_stats.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Merging Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Merge with Fear and Greed Index\n",
    "merged_df = pd.merge(daily_stats, fg[['date', 'value', 'classification']], on='date', how='inner')\n",
    "merged_df['sentiment_regime'] = merged_df['classification']\n",
    "print(f\"Merged Data: {merged_df.shape}\")\n",
    "print(merged_df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Analysis Phase 1: Performance vs Sentiment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# PnL Distribution by Sentiment\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.boxplot(x='classification', y='daily_pnl', data=merged_df, showfliers=False)\n",
    "plt.title('Daily PnL Distribution by Sentiment Regime')\n",
    "plt.xticks(rotation=45)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Win Rate by Sentiment\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.barplot(x='classification', y='win_rate', data=merged_df, errorbar=None)\n",
    "plt.title('Average Win Rate by Sentiment Regime')\n",
    "plt.ylim(0, 1)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Analysis Phase 2: Behavioral Changes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Trade Frequency vs Sentiment\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.barplot(x='classification', y='trade_count', data=merged_df)\n",
    "plt.title('Average Daily Trade Count by Sentiment')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Trade Size vs Sentiment\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.barplot(x='classification', y='avg_trade_size', data=merged_df)\n",
    "plt.title('Average Trade Size by Sentiment')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Trader Segmentation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define Segments\n",
    "merged_df['size_segment'] = pd.qcut(merged_df['avg_trade_size'], 3, labels=['Low', 'Medium', 'High'])\n",
    "merged_df['freq_segment'] = pd.qcut(merged_df['trade_count'], 3, labels=['Low', 'Medium', 'High'])\n",
    "\n",
    "# Analyze Sensitivity\n",
    "segment_pnl = merged_df.groupby(['size_segment', 'classification'])['daily_pnl'].mean().unstack()\n",
    "print(segment_pnl)\n",
    "segment_pnl.plot(kind='bar', figsize=(12, 6))\n",
    "plt.title('PnL by Trader Size Segment and Sentiment')\n",
    "plt.ylabel('Average Daily PnL')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Predictive Modeling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "\n",
    "# Target: Profitable Day (1) or Not (0)\n",
    "merged_df['is_profitable'] = (merged_df['daily_pnl'] > 0).astype(int)\n",
    "\n",
    "# Features\n",
    "features = ['value', 'trade_count', 'avg_trade_size', 'long_ratio', 'total_volume']\n",
    "X = merged_df[features].fillna(0)\n",
    "y = merged_df['is_profitable']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "clf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "y_pred = clf.predict(X_test)\n",
    "print(\"Model Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(classification_report(y_test, y_pred))\n",
    "\n",
    "# Feature Importance\n",
    "importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)\n",
    "print(\"\\nFeature Importances:\")\n",
    "print(importances)\n",
    "importances.plot(kind='bar', title='Feature Importance for Profitability Prediction')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save processed data for Dashboard\n",
    "merged_df.to_csv('processed_data.csv', index=False)\n",
    "print(\"Processed data saved to processed_data.csv\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.x"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open('analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
