# Maritime Macro & Equity Quant Dashboard

[![Open Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://PASTE-YOUR-APP-URL.streamlit.app)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**[▶ Try the live dashboard, no installation needed](https://PASTE-YOUR-APP-URL.streamlit.app)**

A quantitative research dashboard for the shipping sector. It connects physical freight markets, bunker fuel costs and listed shipping equities through a stationarity aware analytics pipeline, then lets you backtest timing strategies on the relationships it finds. Built as the maritime extension of my [Macro-quant-dashboard-demo](https://github.com/sebastianSEEB/Macro-quant-dashboard-demo).

<!-- Add a preview GIF here after recording one:
![Dashboard preview](screenshots/preview.gif)
-->

## What it answers

1. **How tight is the link right now?** Rolling correlation matrices at 30, 90, 180 and 365 day windows show whether shipping stocks are tracking freight rates or have decoupled.
2. **Who moves first?** Lead-lag cross correlation over a -15 to +15 day window identifies whether physical freight moves predict equity returns or the stock market prices rates in advance.
3. **Is there a predictive signal?** Multi-horizon regressions (1, 3, 5, 10, 22 day forward returns) with Newey-West robust errors test whether today's freight or fuel change carries statistically significant information about future returns.
4. **Would a simple strategy have worked?** A walk-forward Strategy Simulator backtests momentum timing and an ML pairs reversion engine that retrains a rolling regression every day, with signals lagged one day so there is zero look-ahead bias.

## The six tabs

| Tab | What it does |
|---|---|
| Macro overlay | Dual-axis chart of a freight or fuel driver against the equities, rebased for cross-currency comparison |
| Correlation structure | Annotated Pearson/Spearman heatmap with a window slider, plus rolling pairwise correlation over time |
| Lead-lag (CCF) | Cross correlation across 31 lags with per-lag significance bands and the optimal lag highlighted |
| Predictive horizons | Regression matrix across five forward horizons, color coded by Newey-West significance |
| Methodology | ADF stationarity audit on every series plus plain language method notes |
| Strategy Simulator | Walk-forward momentum and ML statistical arbitrage backtests vs buy and hold |

## Architecture

```
maritime-quant-dashboard/
├── app_maritime.py          # Streamlit entry point, layout and state
├── data/
│   └── maritime_loader.py   # ETL, fallback chains, 12h cache, feed status
├── analytics/
│   └── quant_engine.py      # ADF audit, correlations, CCF, NW regressions
├── ui/
│   └── components.py        # Themed Plotly figures, styled tables, desk notes
├── .streamlit/config.toml   # Corporate navy theme
└── requirements.txt
```

The quant engine has zero Streamlit dependency, so it imports cleanly into a notebook or test suite. The lead-lag analyzer was validated against synthetic data with a planted 2 day signal, which it recovered exactly.

## Asset universe

| Role | Asset | Ticker | Notes |
|---|---|---|---|
| Dry bulk freight | BDI proxy | BDRY | Breakwave Dry Bulk Shipping ETF |
| Tanker freight | BDTI proxy | BWET | Breakwave Tanker Shipping ETF, history from late 2023 |
| Bunker fuel | VLSFO/MGO proxy | HO=F, fallback BZ=F | NY Harbor ULSD futures |
| Crude benchmark | Brent | BZ=F | |
| Equity | Frontline plc | FRO | Crude tankers, heavy spot exposure |
| Equity | A.P. Moller-Maersk B | MAERSK-B.CO, fallback AMKBY | Container liner |
| Equity | Hafnia Ltd | HAFNI.OL | Product tankers |
| Equity | Stolt-Nielsen Ltd | SNI.OL | Chemical parcel tankers, COA heavy |

### Why proxies instead of Baltic Exchange data

Baltic Exchange indices (BDI, BDTI) are licensed, paywalled data. Scraping them through aggregators violates terms of service and produces a brittle pipeline. This project deliberately uses exchange traded freight ETFs instead: legal to use, free to fetch, liquid and investable, which makes any signal found in them economically meaningful. The tradeoff (futures roll costs and fees mean the ETFs track rate direction rather than spot levels) is documented in the Methodology tab.

## Methodology in brief

- **Stationarity first.** Equities become log returns, freight and fuel become daily percentage changes, and an Augmented Dickey-Fuller audit flags anything that fails to reject the unit root null. Raw trending prices never enter a correlation or regression.
- **Honest inference.** Overlapping h-day forward returns are autocorrelated by construction, so all predictive regressions use Newey-West HAC standard errors.
- **No look-ahead bias.** Every backtest signal is lagged one day: yesterday's information decides today's exposure. The ML pairs engine retrains its rolling regression daily on a strictly trailing window.
- **Graceful degradation.** Every asset has a fallback ticker chain, data is cached for 12 hours, failed feeds show a banner instead of crashing, and thin samples trigger explicit low-confidence warnings.

## Run it locally

```bash
git clone https://github.com/sebastianSEEB/maritime-quant-dashboard.git
cd maritime-quant-dashboard
python -m venv venv
venv\Scripts\activate          # mac/linux: source venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app_maritime.py
```

First load fetches three years of daily data for eight assets and caches it for 12 hours.

## Limitations, stated honestly

- ETF proxies embed roll yield and fees and start later than the underlying indices; BWET in particular limits long tanker lookbacks.
- Backtests ignore transaction costs, slippage and taxes, so strategy outperformance is optimistic by design.
- Oslo, Copenhagen and New York listings settle in different currencies; returns based analytics absorb most of this, but FX moves remain inside NOK and DKK return series.
- This is an educational research tool. Nothing in it is investment advice or a recommendation to trade any security.

## About

Built by [Sebastian Kleven](https://sebastianseeb.github.io), BBA student at BI Norwegian Business School specializing in shipping, currently on exchange at NUCB in Japan. Python is the hobby, shipping and finance are the career.

## License

MIT
