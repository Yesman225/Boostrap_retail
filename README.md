# European Portfolio Co-Pilot

A Streamlit app that helps retail investors explore large European stocks, understand the risk/return trade-off, and design a portfolio in minutes. The experience is guided:

1. **Choose your universe** – pick indices, filter for listing age or currency, and select the companies you like.
2. **Shape your risk & return** – move a slider along the efficient frontier to balance stability and growth.
3. **Review the mix** – see expected return, volatility, weights, and historical performance with clean charts.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Project layout

```
app.py                    # Streamlit entry point with the guided flow
src/
├── config.py             # Constants, theme colours, cache defaults
├── data/                 # Index scraping, filtering, price downloads
├── portfolio/            # Optimisation, efficient frontier, metrics
├── plotting/             # Beautiful Matplotlib charts
└── ui/                   # Session-state helpers and widgets
legacy/                   # Original monolithic scripts (kept for reference)
requirements.txt          # Python dependencies
README.md                 # This file
```

## Key features

- Batch scraping of CAC 40, DAX, AEX, IBEX and FTSE MIB constituents
- Currency and history filters built on top of `yfinance`
- Modern portfolio optimisation (minimum variance, max return, frontier sweep)
- Responsive plots (donut allocation, price evolution, highlighted frontier point)
- Extensive caching to keep the UX snappy across page interactions

## Next ideas

- Add an onboarding modal explaining risk and return concepts in plain language
- Convert optimisation routines to Numba for even faster frontier sweeps
- Offer a "€ investment" slider to translate returns into money terms

Enjoy exploring, and feel free to tailor the experience for your own audience!
