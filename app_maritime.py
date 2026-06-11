"""Maritime Macro & Equity Quant Dashboard.

Main Streamlit entry point. Wires the cached data pipeline
(data/maritime_loader.py) into the analytics engine
(analytics/quant_engine.py) and renders everything through the shared
component library (ui/components.py).

Run with:
    streamlit run app_maritime.py
"""

from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import streamlit as st

from analytics import quant_engine as qe
from data import maritime_loader as loader
from ui import components as ui

st.set_page_config(
    page_title="Maritime Macro & Equity Quant Dashboard",
    page_icon=":anchor:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Maritime Quant")
st.sidebar.caption("Freight, fuel and shipping equities. Signals, not noise.")

today = dt.date.today()
default_start = today - dt.timedelta(days=365 * 3)
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(default_start, today),
    max_value=today,
)
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

prices, statuses = loader.load_price_panel(str(start_date), str(end_date))

if prices.empty:
    st.error(
        "No market data could be loaded. Check your internet connection, "
        "then rerun. All feeds failed."
    )
    ui.render_feed_status(statuses)
    st.stop()

available = set(prices.columns)
equities = [k for k in loader.EQUITY_KEYS if k in available]
freight = [k for k in loader.FREIGHT_KEYS if k in available]
bunker = [k for k in loader.BUNKER_KEYS if k in available]
macro = freight + bunker

driver_options = {loader.ASSETS[k].name: k for k in macro}
equity_options = {loader.ASSETS[k].name: k for k in equities}

selected_driver_name = st.sidebar.selectbox("Macro driver", list(driver_options))
selected_driver = driver_options[selected_driver_name]

selected_equity_names = st.sidebar.multiselect(
    "Shipping equities",
    list(equity_options),
    default=list(equity_options),
)
selected_equities = [equity_options[n] for n in selected_equity_names]
if not selected_equities:
    st.sidebar.warning("Select at least one equity.")
    st.stop()

corr_method = st.sidebar.radio("Correlation method", ("pearson", "spearman"), horizontal=True)

# ---------------------------------------------------------------------------
# Header, feed status, overview
# ---------------------------------------------------------------------------

st.title("Maritime Macro & Equity Quant Dashboard")
st.caption(
    "Freight rate proxies, bunker fuel and listed shipping equities through "
    "one stationarity aware pipeline: rolling correlation structure, "
    "lead-lag cross correlation, and multi-horizon predictive regressions "
    "with Newey-West robust inference."
)

ui.render_feed_status(statuses)
ui.render_proxy_notes()

returns = qe.transform_to_returns(prices, loader.EQUITY_KEYS, loader.MACRO_KEYS)
summary = loader.latest_summary(prices)

st.subheader("Sector overview")
ui.render_overview_cards(summary, macro)
ui.render_overview_cards(summary, equities)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overlay, tab_corr, tab_ccf, tab_pred, tab_method, tab_strat = st.tabs(
    [
        "Macro overlay",
        "Correlation structure",
        "Lead-lag (CCF)",
        "Predictive horizons",
        "Methodology",
        "Strategy Simulator",
    ]
)

with tab_overlay:
    fig = ui.dual_axis_overlay(prices, selected_driver, selected_equities)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Left axis: selected macro driver in its own units. Right axis: "
        "equities, rebased to 100 at the window start when more than one is "
        "selected so NOK, DKK and USD names stay comparable."
    )
    st.info(
        f"**What this data shows:** You are comparing the raw price trajectory of **{selected_driver_name}** "
        f"against the selected shipping equities ({', '.join(selected_equity_names)}). "
        "This view helps visually identify long-term macroeconomic regimes and whether the equities "
        "generally trend alongside the physical market over multi-year horizons."
    )

with tab_corr:
    window = st.select_slider(
        "Trailing window (trading days)", options=(30, 90, 180, 365), value=90
    )
    corr_keys = macro + selected_equities
    matrix = qe.trailing_correlation(returns[corr_keys], window, corr_method)
    st.plotly_chart(
        ui.heatmap_figure(matrix, window, corr_method), use_container_width=True
    )

    st.markdown("**Rolling pairwise correlation**")
    pair_equity_name = st.selectbox(
        "Equity for rolling view",
        selected_equity_names,
        key="rolling_pair_equity",
    )
    pair_key = equity_options[pair_equity_name]
    rolling = qe.rolling_pair_correlation(returns, selected_driver, pair_key)
    if rolling.empty:
        st.warning("Not enough overlapping history for a rolling window on this pair.")
    else:
        st.plotly_chart(
            ui.rolling_correlation_figure(rolling, selected_driver, pair_key),
            use_container_width=True,
        )
    ui.render_domain_note(pair_key)
    st.info(
        f"**What this data shows:** The line chart tracks the **{window}-day rolling {corr_method} correlation** "
        f"between **{selected_driver_name}** and **{pair_equity_name}**. When the line is high, "
        "the two assets are moving in lockstep. Sudden drops indicate a decoupling, which might "
        "be caused by equity-specific news or broader market sell-offs ignoring maritime fundamentals."
    )

with tab_ccf:
    ccf_equity_name = st.selectbox("Equity", selected_equity_names, key="ccf_equity")
    ccf_equity = equity_options[ccf_equity_name]
    result = qe.cross_correlation(returns, selected_driver, ccf_equity)
    if result.low_sample:
        st.warning(
            f"Short overlapping sample (min {int(result.table['n'].min())} obs per lag). "
            "Treat the bars as indicative, not tradeable."
        )
    st.plotly_chart(
        ui.ccf_figure(result.table, result.optimal_lag, selected_driver, ccf_equity),
        use_container_width=True,
    )
    direction = (
        "physical rates lead the equity"
        if result.optimal_lag > 0
        else "the equity leads physical rates"
        if result.optimal_lag < 0
        else "co-movement is contemporaneous"
    )
    st.markdown(
        f"Optimal lag **{result.optimal_lag:+d} days** "
        f"(rho = {result.optimal_corr:.4f}): {direction}."
    )
    ui.render_domain_note(ccf_equity)
    st.info(
        f"**What this data shows:** By shifting **{selected_driver_name}** forward and backward in time, "
        f"we check who moves first. The optimal lag of {result.optimal_lag} days suggests that "
        f"{direction}. If the lag is positive, changes in the macro driver today ripple into **{ccf_equity_name}**'s price later."
    )

with tab_pred:
    st.markdown(
        "Forward equity log return over each horizon regressed on today's "
        "freight change and bunker change. Newey-West HAC errors handle the "
        "overlap induced autocorrelation."
    )
    st.latex(
        r"R_{t+h} = \alpha + \beta_1\,\Delta\%FR_t + \beta_2\,\Delta\%Bunker_t"
        r" + \varepsilon_t,\qquad h \in \{1,3,5,10,22\}"
    )
    pred_equity_name = st.selectbox("Equity", selected_equity_names, key="pred_equity")
    pred_equity = equity_options[pred_equity_name]

    bunker_driver = "bunker" if "bunker" in available else ("brent" if "brent" in available else None)
    drivers = [selected_driver] if selected_driver in freight else freight[:1]
    if bunker_driver and bunker_driver not in drivers:
        drivers.append(bunker_driver)
    drivers = [d for d in drivers if d in available]

    table = qe.predictive_horizon_table(returns, pred_equity, drivers)
    if table.empty:
        st.warning("Not enough overlapping observations to estimate the horizon matrix.")
    else:
        st.dataframe(
            ui.regression_styler(table, drivers), use_container_width=True
        )
        st.caption(
            "Green: p < 0.05 under Newey-West. Amber: p < 0.10. Blue gradient "
            "tracks adjusted R squared across horizons."
        )
    ui.render_domain_note(pred_equity)
    st.info(
        f"**What this data shows:** This matrix tests if today's moves in our drivers can predict "
        f"the *future* returns of **{pred_equity_name}** across different timeframes (from 1 day to 22 days). "
        "A statistically significant beta at a specific horizon means the equity market takes time to fully "
        "price in the physical freight or bunker data."
    )

with tab_method:
    st.subheader("Stationarity audit (ADF on transformed series)")
    adf = qe.adf_table(returns[macro + equities])
    st.dataframe(
        adf.style.format(
            {"adf_stat": "{:.2f}", "p_value": "{:.4f}", "n_obs": "{:,.0f}"}
        ),
        use_container_width=True,
    )
    failed = adf.index[~adf["stationary"]].tolist()
    if failed:
        st.warning(
            "Series failing the ADF unit root rejection at 5%: "
            + ", ".join(failed)
            + ". Interpret regressions on these with caution."
        )
    else:
        st.success("All transformed series reject the unit root null at 5%.")

    st.subheader("Method notes")
    st.markdown(
        """
- **Transforms.** Equities use log returns. Freight and bunker proxies use simple daily percentage changes. 
- **Correlation.** Pearson measures linear co-movement; Spearman ranks are robust to outliers.
- **Lead-lag CCF.** corr(driver_t, equity_t+k) for k in [-15, +15].
- **Predictive OLS.** Overlapping h-day forward returns use Newey-West (HAC) covariance to prevent false positives.
- **Disclaimer.** Educational research tool. Nothing here is investment advice.
        """
    )
    st.info("**What this data shows:** The stationarity audit verifies the mathematical validity of the previous tabs. Running predictive models on non-stationary data (data that trends indefinitely) creates 'spurious correlations'. By checking the Augmented Dickey-Fuller (ADF) stats, we ensure our models are built on sound, mean-reverting daily changes.")

with tab_strat:
    st.subheader("Strategy Simulator (Walk-Forward)")
    st.markdown("Test simple timing strategies. Signals are generated using **yesterday's** data to dictate **today's** market exposure, enforcing a strict point-in-time backtest with zero look-ahead bias.")
    
    strat_family = st.radio(
        "Strategy Engine", 
        ["Momentum (Trend Following)", "Pairs Reversion (ML Statistical Arbitrage)"], 
        horizontal=True
    )

    st.markdown("---")

    if strat_family == "Momentum (Trend Following)":
        st.latex(r"R_{strategy, t} = \text{Signal}_{t-1} \times R_{asset, t}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            trade_equity_name = st.selectbox("Equity to Trade", selected_equity_names, key="strat_eq")
            trade_equity = equity_options[trade_equity_name]
        with col2:
            strat_type = st.selectbox("Signal Type", ["Macro Momentum", "Equity Momentum"], help="Macro uses the driver to time the equity. Equity uses its own price.")
        with col3:
            lookback = st.number_input("Signal Lookback (Days)", min_value=1, max_value=252, value=20)
            
        signal_target = selected_driver if strat_type == "Macro Momentum" else trade_equity
        
        # 1 if cumulative return over lookback is positive, else 0 (cash)
        rolling_sum = returns[signal_target].rolling(lookback).sum()
        raw_signal = rolling_sum.apply(lambda x: 1 if x > 0 else 0)
        
        # SHIFT SIGNAL BY 1 DAY TO AVOID LOOK-AHEAD BIAS
        executable_signal = raw_signal.shift(1)
        strat_returns = executable_signal * returns[trade_equity]
        
        valid_data = strat_returns.dropna()
        buy_hold_returns = returns[trade_equity].loc[valid_data.index]
        
        cum_strat = (1 + valid_data).cumprod() * 100
        cum_bh = (1 + buy_hold_returns).cumprod() * 100
        
        fig_strat = go.Figure()
        fig_strat.add_trace(go.Scatter(x=cum_strat.index, y=cum_strat, name=f"{strat_type} Strategy", line=dict(color='green')))
        fig_strat.add_trace(go.Scatter(x=cum_bh.index, y=cum_bh, name="Buy & Hold", line=dict(color='gray', dash='dot')))
        fig_strat.update_layout(title="Momentum Strategy vs. Buy & Hold ($100 Starting Capital)", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_strat, use_container_width=True)
        
        st.info(f"**What this data shows:** We are backtesting a rule where you hold **{trade_equity_name}** only when the previous {lookback} days of **{signal_target}** returns are positive. Otherwise, you hold cash (0% return).")

    elif strat_family == "Pairs Reversion (ML Statistical Arbitrage)":
        st.markdown("Trades **Asset A** when its price drops unusually low relative to its ML-predicted baseline against **Asset B**.")
        
        st.latex(r"\text{Spread}_t = \text{Price}_{A,t} - (\alpha_t + \beta_t \cdot \text{Price}_{B,t})")

        col1, col2, col3, col4 = st.columns(4)
        
        all_options = {**equity_options, **driver_options}
        
        with col1:
            asset_a_name = st.selectbox("Asset A (Trade Target)", selected_equity_names, key="pairs_a")
            asset_a = equity_options[asset_a_name]
        with col2:
            asset_b_name = st.selectbox("Asset B (Benchmark Pair)", list(all_options.keys()), index=0, key="pairs_b")
            asset_b = all_options[asset_b_name]
        with col3:
            lookback = st.number_input("ML Training Window (Days)", min_value=15, max_value=252, value=60, step=5)
        with col4:
            z_threshold = st.number_input("Entry Z-Score", max_value=-0.5, value=-1.5, step=0.1)

        if asset_a == asset_b:
            st.warning("Please select two different assets to pair against each other.")
        else:
            # --- ML WALK-FORWARD ENGINE ---
            df_ml = prices[[asset_a, asset_b]].dropna()
            
            spreads = []
            dates = []
            
            for i in range(lookback, len(df_ml)):
                window_data = df_ml.iloc[i - lookback : i]
                current_day = df_ml.iloc[i]
                
                X_train = window_data[[asset_b]].values
                y_train = window_data[asset_a].values
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                X_today = np.array([[current_day[asset_b]]])
                pred_a = model.predict(X_today)[0]
                
                actual_a = current_day[asset_a]
                spread_val = actual_a - pred_a
                
                spreads.append(spread_val)
                dates.append(df_ml.index[i])
                
            ml_spread = pd.Series(index=dates, data=spreads)
            
            roll_spread_mean = ml_spread.rolling(lookback).mean()
            roll_spread_std = ml_spread.rolling(lookback).std()
            z_score = (ml_spread - roll_spread_mean) / roll_spread_std
            
            # --- SIGNAL STATE MACHINE ---
            buy_condition = z_score < z_threshold
            sell_condition = z_score > 0
            
            raw_signal = pd.Series(index=z_score.index, data=0)
            current_position = 0
            
            for date in z_score.index:
                if pd.isna(z_score[date]):
                    continue
                if buy_condition[date]:
                    current_position = 1
                elif sell_condition[date]:
                    current_position = 0
                raw_signal[date] = current_position

            executable_signal = raw_signal.shift(1)

            strat_returns = executable_signal * returns[asset_a]
            valid_data = strat_returns.dropna()
            buy_hold_returns = returns[asset_a].loc[valid_data.index]

            cum_strat = (1 + valid_data).cumprod() * 100
            cum_bh = (1 + buy_hold_returns).cumprod() * 100

            # --- PLOTTING ---
            fig_pairs = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig_pairs.add_trace(go.Scatter(x=cum_strat.index, y=cum_strat, name="ML Pairs Strategy", line=dict(color='green')), row=1, col=1)
            fig_pairs.add_trace(go.Scatter(x=cum_bh.index, y=cum_bh, name=f"Buy & Hold {asset_a_name}", line=dict(color='gray', dash='dot')), row=1, col=1)
            
            fig_pairs.add_trace(go.Scatter(x=z_score.index, y=z_score, name="ML Residual Z-Score", line=dict(color='purple', width=1)), row=2, col=1)
            fig_pairs.add_hline(y=z_threshold, line_dash="dash", line_color="red", row=2, col=1)
            fig_pairs.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

            fig_pairs.update_layout(title="ML Statistical Arbitrage vs. Buy & Hold", template="plotly_white", hovermode="x unified", height=600)
            st.plotly_chart(fig_pairs, use_container_width=True)

            st.info(f"**What this data shows:** Instead of a fixed historical ratio, a **Rolling Ordinary Least Squares (OLS)** model learns the dynamic relationship between your assets every single day. The purple line represents structural deviations from this machine learning baseline. When it breaks below {z_threshold}, the model signals that **{asset_a_name}** is mispriced relative to current macroeconomic realities driven by **{asset_b_name}**.")
