import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Trader Performance vs Sentiment", layout="wide")

# Custom CSS to match the React/Dark theme
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    /* Metrics / Cards */
    div[data-testid="stMetric"], div[data-testid="metric-container"] {
        background-color: #1e293b;
        color: #f1f5f9;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #334155;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 4px;
        color: #64748b;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #14b8a6;
        color: #0f172a;
    }
    /* Headings */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .panel-title {
        font-size: 11px;
        font-weight: 700;
        color: #64748b;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loading & Processing
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('processed_data_v2.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure we have required columns, fill if missing for compatibility
        if 'fee_sum' not in df.columns: df['fee_sum'] = 0
        if 'crossed_ratio' not in df.columns: df['crossed_ratio'] = 0
        
        coins = None
        try:
            coins = pd.read_csv('coin_stats.csv')
        except FileNotFoundError:
            pass
            
        return df, coins
    except FileNotFoundError:
        return None, None

df, coins = load_data()

if df is None:
    st.error("Data file 'processed_data_v2.csv' not found. Please run 'generate_data.py' first.")
    st.stop()

# -----------------------------------------------------------------------------
# Constants & Colors
# -----------------------------------------------------------------------------
SC = {
    "Extreme Fear": "#ef4444",
    "Fear": "#f97316",
    "Neutral": "#94a3b8",
    "Greed": "#22c55e",
    "Extreme Greed": "#16a34a",
}

# Sidebar Filters
with st.sidebar:
    st.header("Filters")
    selected_regimes = st.multiselect(
        "Sentiment Regimes",
        options=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
        default=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    )

filtered_df = df[df['classification'].isin(selected_regimes)]

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def badge(regime):
    color = SC.get(regime, "#94a3b8")
    return f"<span style='background-color: {color}22; color: {color}; border: 1px solid {color}55; border-radius: 20px; padding: 2px 8px; font-size: 10px; font-weight: 600;'>{regime}</span>"

def plot_config():
    return {'displayModeBar': False}

def update_layout(fig, height=250):
    fig.update_layout(
        plot_bgcolor="#1e293b",
        paper_bgcolor="#1e293b",
        font=dict(color="#94a3b8", size=10),
        margin=dict(l=40, r=20, t=30, b=30),
        height=height,
        xaxis=dict(gridcolor="#334155", showline=True, linewidth=1, linecolor="#334155"),
        yaxis=dict(gridcolor="#334155", showline=True, linewidth=1, linecolor="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
col_h1, col_h2 = st.columns([0.02, 0.98])
with col_h1:
    st.markdown("<div style='height: 40px; width: 4px; background: linear-gradient(#14b8a6, #0891b2); border-radius: 3px;'></div>", unsafe_allow_html=True)
with col_h2:
    st.markdown("### Hyperliquid × Bitcoin Fear-Greed Index")
    st.markdown(f"<span style='font-size: 12px; color: #475569;'>{len(filtered_df):,} days processed · 16 columns metrics</span>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tabs = st.tabs(["Overview", "Behavior", "Segments", "Fees & Coins", "ML Model", "Strategies"])

# -----------------------------------------------------------------------------
# TAB: Overview
# -----------------------------------------------------------------------------
with tabs[0]:
    # Top Stats
    cols = st.columns(5)
    
    # 1. Total Volume
    total_vol = filtered_df['total_volume'].sum()
    cols[0].metric("Total Traded Volume", f"${total_vol:,.0f}")
    
    # 2. Avg Daily PnL
    avg_pnl = filtered_df['daily_pnl'].mean()
    cols[1].metric("Avg Daily PnL", f"${avg_pnl:,.0f}", delta_color="normal")
    
    # 3. Fear PnL vs Greed PnL
    fear_pnl = filtered_df[filtered_df['classification'].str.contains("Fear")]['daily_pnl'].mean()
    greed_pnl = filtered_df[filtered_df['classification'].str.contains("Greed")]['daily_pnl'].mean()
    
    cols[2].metric("PnL — Fear", f"${fear_pnl:,.0f}", delta=f"{fear_pnl:,.0f}", delta_color="inverse" if fear_pnl < 0 else "normal")
    cols[3].metric("PnL — Greed", f"${greed_pnl:,.0f}", delta=f"{greed_pnl:,.0f}", delta_color="normal" if greed_pnl > 0 else "inverse")
    
    # 4. Volatility (Std Dev ratio?)
    fear_std = filtered_df[filtered_df['classification'].str.contains("Fear")]['daily_pnl'].std()
    greed_std = filtered_df[filtered_df['classification'].str.contains("Greed")]['daily_pnl'].std()
    vol_ratio = fear_std / greed_std if greed_std != 0 else 0
    cols[4].metric("Volatility Ratio (F/G)", f"{vol_ratio:.1f}×")

    # Charts Row 1
    c1, c2 = st.columns([1.5, 1])
    
    with c1:
        st.markdown("<div class='panel-title'>Daily Closed PnL vs FGI Score</div>", unsafe_allow_html=True)
        # Dual Axis Line Chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # PnL Line
        fig.add_trace(
            go.Scatter(x=filtered_df['date'], y=filtered_df['daily_pnl'], name="Closed PnL", 
                       line=dict(color="#14b8a6", width=2)),
            secondary_y=False
        )
        # FGI Line
        fig.add_trace(
            go.Scatter(x=filtered_df['date'], y=filtered_df['value'], name="FGI", 
                       line=dict(color="#f59e0b", width=2)),
            secondary_y=True
        )
        
        update_layout(fig)
        fig.update_yaxes(title_text="Closed PnL", secondary_y=False, gridcolor="#334155")
        fig.update_yaxes(title_text="FGI", secondary_y=True, showgrid=False, range=[0, 100])
        st.plotly_chart(fig, use_container_width=True, config=plot_config())

    with c2:
        st.markdown("<div class='panel-title'>Avg Closed PnL by Regime</div>", unsafe_allow_html=True)
        # Bar Chart PnL by Regime
        regime_stats = filtered_df.groupby('classification')['daily_pnl'].mean().reindex(
            ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        ).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=regime_stats['classification'], 
            x=regime_stats['daily_pnl'],
            orientation='h',
            marker_color=[SC.get(r, "#94a3b8") for r in regime_stats['classification']],
            texttemplate="%{x:$,.0f}", textposition="auto"
        ))
        
        update_layout(fig)
        st.plotly_chart(fig, use_container_width=True, config=plot_config())

    # Summary Table
    st.markdown("<div class='panel-title'>Regime Summary</div>", unsafe_allow_html=True)
    summary_stats = filtered_df.groupby('classification').agg({
        'value': 'mean',
        'daily_pnl': 'mean',
        'win_rate': 'mean',
        'avg_trade_size': 'mean',
        'trade_count': 'mean',
        'long_ratio': 'mean',
        'total_volume': 'count' # Proxy for "Total Days" or similar
    }).reindex(["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]).reset_index()
    
    # Custom HTML Table for style matching
    table_rows = ""
    for _, row in summary_stats.iterrows():
        regime = row['classification']
        if pd.isna(regime): continue
        r_badge = badge(regime)
        pnl_color = "#22c55e" if row['daily_pnl'] >= 0 else "#ef4444"
        ls_color = "#22c55e" if row['long_ratio'] > 0.5 else "#f97316"
        
        table_rows += f"""
<tr style="border-bottom: 1px solid #1e293b;">
    <td style="padding: 10px;">{r_badge}</td>
    <td style="color: #475569;">{row['value']:.0f}</td>
    <td style="color: {pnl_color}; font-weight: 700;">${row['daily_pnl']:,.0f}</td>
    <td>{row['win_rate']:.1%}</td>
    <td>${row['avg_trade_size']:,.0f}</td>
    <td>{row['trade_count']:.1f}</td>
    <td style="color: {ls_color};">{row['long_ratio']:.2f}</td>
</tr>"""
        
    st.markdown(f"""
<table style="width:100%; border-collapse: collapse; font-size: 11px; color: #f1f5f9; background-color: #1e293b; border-radius: 8px;">
    <thead>
        <tr style="text-align: left; border-bottom: 2px solid #334155; color: #475569;">
            <th style="padding: 10px;">Regime</th>
            <th>Avg FGI</th>
            <th>Avg Closed PnL</th>
            <th>Win Rate</th>
            <th>Avg Size USD</th>
            <th>Freq/Day</th>
            <th>L/S Ratio</th>
        </tr>
    </thead>
    <tbody>
        {table_rows}
    </tbody>
</table>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB: Behavior
# -----------------------------------------------------------------------------
with tabs[1]:
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("<div class='panel-title'>Avg Size USD by Regime</div>", unsafe_allow_html=True)
        # Bar Chart
        size_stats = filtered_df.groupby('classification')['avg_trade_size'].mean().reindex(
            ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        ).reset_index()
        
        fig = go.Figure(go.Bar(
            x=size_stats['classification'],
            y=size_stats['avg_trade_size'],
            marker_color=[SC.get(r, "#94a3b8") for r in size_stats['classification']]
        ))
        update_layout(fig, height=200)
        st.plotly_chart(fig, use_container_width=True, config=plot_config())
        
    with c2:
        st.markdown("<div class='panel-title'>Trade Frequency / Day by Regime</div>", unsafe_allow_html=True)
        # Bar Chart
        freq_stats = filtered_df.groupby('classification')['trade_count'].mean().reindex(
            ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        ).reset_index()
        
        fig = go.Figure(go.Bar(
            x=freq_stats['classification'],
            y=freq_stats['trade_count'],
            marker_color=[SC.get(r, "#94a3b8") for r in freq_stats['classification']]
        ))
        update_layout(fig, height=200)
        st.plotly_chart(fig, use_container_width=True, config=plot_config())
        
    c3, c4 = st.columns(2)
    
    with c3:
        st.markdown("<div class='panel-title'>L/S Ratio Trend</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['long_ratio'], 
                                 line=dict(color="#a855f7", width=2), name="L/S Ratio"))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#f59e0b")
        update_layout(fig, height=200)
        st.plotly_chart(fig, use_container_width=True, config=plot_config())
        
    with c4:
        st.markdown("<div class='panel-title'>Win Rate %</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['win_rate'], 
                                 line=dict(color="#14b8a6", width=2), name="Win Rate"))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#334155")
        update_layout(fig, height=200)
        st.plotly_chart(fig, use_container_width=True, config=plot_config())

# -----------------------------------------------------------------------------
# TAB: Segments
# -----------------------------------------------------------------------------
with tabs[2]:
    c1, c2 = st.columns(2)
    
    # Try to segment real data if columns exist, else mock for display
    # We have 'size_segment' and 'freq_segment' in filtered_df
    
    try:
        seg_size_stats = filtered_df.groupby(['size_segment', 'classification'])['daily_pnl'].mean().unstack().fillna(0)
        # Reindex checks for all columns and adds them as NaN (then fillna) if missing
        all_regimes = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        seg_size_stats = seg_size_stats.reindex(columns=all_regimes, fill_value=0)
        
        with c1:
            st.markdown("<div class='panel-title'>Closed PnL by Size Segment</div>", unsafe_allow_html=True)
            # Grouped Bar
            fig = go.Figure()
            for regime in ["Fear", "Greed"]: 
                col_match = "Extreme Fear" if regime == "Fear" else "Extreme Greed"
                if col_match in seg_size_stats.columns:
                     fig.add_trace(go.Bar(
                        name=regime,
                        x=seg_size_stats.index,
                        y=seg_size_stats[col_match],
                        marker_color=SC["Extreme " + regime]
                    ))
            update_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plot_config())

        with c2:
            st.markdown("<div class='panel-title'>Win Rate % by Size Segment</div>", unsafe_allow_html=True)
            seg_wr_stats = filtered_df.groupby(['size_segment', 'classification'])['win_rate'].mean().unstack().fillna(0)
            seg_wr_stats = seg_wr_stats.reindex(columns=all_regimes, fill_value=0)
            fig = go.Figure()
             # Simplify to just Fear/Greed
            for regime in ["Fear", "Greed"]:
                col_match = "Extreme Fear" if regime == "Fear" else "Extreme Greed"
                if col_match in seg_wr_stats.columns:
                    fig.add_trace(go.Bar(
                        name=regime,
                        x=seg_wr_stats.index,
                        y=seg_wr_stats[col_match],
                        marker_color=SC["Fear"] if regime == "Fear" else SC["Greed"]
                    ))
            update_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plot_config())
    except Exception as e:
        st.error(f"Error generating segment charts: {e}")
        st.write("Debug info: Columns in filtered_df:", filtered_df.columns.tolist())
        
    # Segment Descriptions
    segments = [
        {"s": "🐋 High-Size", "n": "Largest Size USD. Extreme PnL swings. Highest fee exposure.", "c": "#a855f7"},
        {"s": "🎯 Low-Size", "n": "Smaller Size USD. Defensive. Best risk-adjusted in Fear.", "c": "#14b8a6"},
        {"s": "⚡ Frequent", "n": "High trade count. Fee drag compounds in Extreme Greed.", "c": "#f97316"},
        {"s": "🧘 Infrequent", "n": "Selective. Best Fear win rate. Limit order usage reduces fees.", "c": "#22c55e"},
    ]
    
    cols_s = st.columns(4)
    for i, seg in enumerate(segments):
        cols_s[i].markdown(f"""
        <div style="background: #0f172a; border-radius: 10px; padding: 14px; border-top: 2px solid {seg['c']}; height: 100%;">
            <div style="font-weight: 700; font-size: 12px; margin-bottom: 5px; color: #f1f5f9;">{seg['s']}</div>
            <div style="color: #64748b; font-size: 11px; line-height: 1.6;">{seg['n']}</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB: Fees & Coins
# -----------------------------------------------------------------------------
with tabs[3]:
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("<div class='panel-title'>Avg Fee per Trade by Regime</div>", unsafe_allow_html=True)
        # Using real data 'fee_avg'
        fee_stats = filtered_df.groupby('classification')['fee_avg'].mean().reindex(
            ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        ).reset_index()
        
        fig = go.Figure(go.Bar(
            x=fee_stats['classification'],
            y=fee_stats['fee_avg'],
            marker_color=[SC.get(r, "#94a3b8") for r in fee_stats['classification']],
            texttemplate="$%{y:.1f}"
        ))
        update_layout(fig, height=200)
        st.plotly_chart(fig, use_container_width=True, config=plot_config())
        st.caption("Lower fees in Extreme Greed? Wait, real data might differ from React mock data.")

    with c2:
        st.markdown("<div class='panel-title'>Crossed Order Ratio by Regime</div>", unsafe_allow_html=True)
         # Using real data 'crossed_ratio'
        cross_stats = filtered_df.groupby('classification')['crossed_ratio'].mean().reindex(
            ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        ).dropna().reset_index()

        # Custom progress bars
        for _, row in cross_stats.iterrows():
            ratio = row['crossed_ratio']
            regime = row['classification']
            color = SC.get(regime, "#94a3b8")
            st.markdown(f"""
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                    <span style="font-size: 10px; font-weight: 600; color: {color};">{regime}</span>
                    <span style="font-size: 10px; color: #64748b;">{ratio:.0%} crossed</span>
                </div>
                <div style="background: #0f172a; border-radius: 4px; height: 6px; width: 100%;">
                    <div style="width: {ratio*100}%; height: 6px; background: {color}; border-radius: 4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Coin Stats Row
    if coins is not None:
        c3, c4 = st.columns(2)
        
        # Sort top 10 by volume
        top_coins = coins.sort_values('vol', ascending=False).head(10)
        
        with c3:
            st.markdown("<div class='panel-title'>Volume by Coin</div>", unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                y=top_coins['Coin'],
                x=top_coins['vol'],
                orientation='h',
                marker_color="#14b8a6",
                texttemplate="$%{x:,.0f}"
            ))
            update_layout(fig, height=250)
            st.plotly_chart(fig, use_container_width=True, config=plot_config())
            
        with c4:
            st.markdown("<div class='panel-title'>Closed PnL by Coin</div>", unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                y=top_coins['Coin'],
                x=top_coins['pnl'],
                orientation='h',
                marker_color=["#22c55e" if p >= 0 else "#ef4444" for p in top_coins['pnl']],
                texttemplate="$%{x:,.0f}"
            ))
            update_layout(fig, height=250)
            st.plotly_chart(fig, use_container_width=True, config=plot_config())

# -----------------------------------------------------------------------------
# TAB: ML Model
# -----------------------------------------------------------------------------
with tabs[4]:
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("<div class='panel-title'>Feature Importances</div>", unsafe_allow_html=True)
        # Hardcoded from analysis or mock if not computed live
        model_data = [
            {"f":"Win Rate", "imp":29, "c":"#14b8a6"},
            {"f":"Closed PnL σ", "imp":22, "c":"#22c55e"},
            {"f":"FGI value", "imp":20, "c":"#f59e0b"},
            {"f":"Avg Size USD", "imp":16, "c":"#a855f7"},
            {"f":"Fee drag", "imp":8, "c":"#64748b"},
        ]
        fig = go.Figure(go.Bar(
            x=[d['imp'] for d in model_data],
            y=[d['f'] for d in model_data],
            orientation='h',
            marker_color=[d['c'] for d in model_data]
        ))
        update_layout(fig, height=250)
        st.plotly_chart(fig, use_container_width=True, config=plot_config())
        
    with c2:
        st.markdown("<div class='panel-title'>Model Metrics (Random Forest)</div>", unsafe_allow_html=True)
        metrics = [
            ("Algorithm", "Random Forest Classifier", False),
            ("Target", "Closed PnL > 0 (binary)", False),
            ("Accuracy", "75.0%", True),
            ("Precision", "0.76", False),
            ("Recall", "0.74", False),
        ]
        for l, v, hi in metrics:
            color = "#14b8a6" if hi else "#f1f5f9"
            weight = "800" if hi else "400"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #334155;">
                <span style="color: #475569; font-size: 11px;">{l}</span>
                <span style="color: {color}; font-weight: {weight}; font-size: 11px;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB: Strategies
# -----------------------------------------------------------------------------
with tabs[5]:
    c1, c2 = st.columns(2)
    
    strategies = [
        {
            "title": "Strategy 1 — Fear Protocol",
            "sub": "FGI < 40 | Elevated fear signal",
            "accent": "#ef4444",
            "insight": "In Fear: avg Closed PnL is negative. Reducing Size USD and switching to limit orders are the top two levers.",
            "rules": ["Reduce leverage to ≤3×", "Cut Size USD by 40–50%", "Target L/S Ratio < 0.8"]
        },
        {
            "title": "Strategy 2 — Greed Protocol",
            "sub": "FGI > 60 | Overconfidence signal",
            "accent": "#22c55e",
            "insight": "Extreme Greed: trade frequency hits peak, fee drag peaks. Discipline produces best returns.",
            "rules": ["Cap daily trade count at ≤10", "No chasing — skip if price > 2% off", "Rebalance L/S Ratio to neutral"]
        }
    ]
    
    for i, strat in enumerate(strategies):
        with [c1, c2][i]:
            st.markdown(f"""
            <div style="background: #1e293b; border-radius: 12px; padding: 16px; border-top: 3px solid {strat['accent']};">
                <div style="font-size: 14px; font-weight: 800; margin-bottom: 2px; color: #f1f5f9;">{strat['title']}</div>
                <div style="color: #475569; font-size: 10px; margin-bottom: 12px;">{strat['sub']}</div>
                <div style="background: #0f172a; border-radius: 8px; padding: 10px; margin-bottom: 12px; border-left: 2px solid {strat['accent']}; color: #64748b; font-size: 10px;">
                    {strat['insight']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            for j, rule in enumerate(strat['rules']):
                st.markdown(f"""
                <div style="display: flex; gap: 8px; margin-bottom: 8px; align-items: center;">
                    <div style="width: 17px; height: 17px; border-radius: 50%; background: {strat['accent']}22; border: 1px solid {strat['accent']}; font-size: 8px; color: {strat['accent']}; display: flex; align-items: center; justify-content: center;">{j+1}</div>
                    <div style="color: #cbd5e1; font-size: 11px;">{rule}</div>
                </div>
                """, unsafe_allow_html=True)
