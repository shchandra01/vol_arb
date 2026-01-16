# Volatility Arbitrage Strategies

A comprehensive Python framework for exploring and implementing volatility arbitrage trading strategies. This repository provides tools for analyzing volatility, pricing options, and executing various volatility trading strategies.

## Overview

Volatility arbitrage exploits differences between implied volatility (from options markets) and realized/historical volatility. This framework includes:

- **Multiple volatility calculation methods** (Historical, Parkinson, Garman-Klass, Yang-Zhang, EWMA)
- **Black-Scholes pricing and Greeks** calculations
- **Four core volatility trading strategies**:
  - Delta-neutral volatility trading
  - Volatility mean reversion
  - Calendar spreads
  - Straddles and strangles
- **Market data fetching** via yfinance
- **Example scripts** demonstrating strategy usage

## Project Structure

```
vol_arb/
├── vol_arb/
│   ├── strategies/
│   │   ├── delta_neutral.py          # Delta-neutral volatility trading
│   │   ├── vol_mean_reversion.py     # Mean reversion strategy
│   │   ├── calendar_spread.py        # Calendar/time spreads
│   │   └── straddle_strangle.py      # Straddle/strangle strategies
│   ├── utils/
│   │   ├── volatility.py             # Volatility calculations
│   │   └── black_scholes.py          # Option pricing and Greeks
│   ├── data/
│   │   └── market_data.py            # Market data fetching
│   └── examples/
│       ├── example_volatility_comparison.py
│       ├── example_straddle_strategy.py
│       └── example_calendar_spread.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shchandra01/vol_arb.git
cd vol_arb
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Compare Implied vs Historical Volatility

Find volatility arbitrage opportunities by comparing IV and HV:

```python
from vol_arb.data.market_data import MarketDataFetcher

fetcher = MarketDataFetcher()
comparison = fetcher.compare_implied_vs_historical('AAPL', hv_window=30)

print(f"Implied Vol: {comparison['implied_vol']*100:.2f}%")
print(f"Historical Vol: {comparison['historical_vol']*100:.2f}%")
print(f"Signal: {comparison['signal']}")
```

### 2. Analyze a Straddle Strategy

```python
from vol_arb.strategies.straddle_strangle import StraddleStrangleStrategy

strategy = StraddleStrangleStrategy()

# Calculate straddle metrics
metrics = strategy.calculate_straddle_metrics(
    S=450.0,      # Stock price
    K=450.0,      # Strike (ATM)
    T=30/365,     # 30 days to expiration
    r=0.05,       # Risk-free rate
    sigma=0.20    # Implied volatility
)

print(f"Total Cost: ${metrics['total_cost']:.2f}")
print(f"Breakeven Move: ±{metrics['breakeven_move_pct']:.1f}%")
print(f"Net Vega: ${metrics['net_vega']:.2f}")
```

### 3. Run Example Scripts

```bash
cd vol_arb/examples
python example_volatility_comparison.py
python example_straddle_strategy.py
python example_calendar_spread.py
```

## Strategies

### 1. Delta-Neutral Volatility Trading

Maintains a delta-neutral position while capturing volatility changes and gamma profits.

**Key Features:**
- Automatic delta hedging and rebalancing
- Gamma scalping P&L tracking
- Opportunity analysis (IV vs HV)

**Use Case:** Buy options when IV < HV, hedge delta, profit from vol increase and gamma scalping.

```python
from vol_arb.strategies.delta_neutral import DeltaNeutralStrategy

strategy = DeltaNeutralStrategy(rebalance_threshold=0.1)

# Analyze opportunity
opportunity = strategy.analyze_opportunity(
    S=450, K=450, T=30/365, r=0.05,
    implied_vol=0.18, historical_vol=0.22, option_type='call'
)

if opportunity['recommendation'] == 'BUY':
    # Enter position and establish hedge
    strategy.enter_position(S=450, K=450, T=30/365, r=0.05,
                          sigma=0.18, option_type='call',
                          quantity=10, option_price=opportunity['price_implied'])
```

### 2. Volatility Mean Reversion

Exploits the tendency of volatility to revert to its historical mean.

**Key Features:**
- Automatic tracking of volatility statistics
- Z-score based entry/exit signals
- Long and short volatility positions

**Use Case:** Buy options when IV is abnormally low, sell when abnormally high.

```python
from vol_arb.strategies.vol_mean_reversion import VolatilityMeanReversionStrategy

strategy = VolatilityMeanReversionStrategy(
    lookback_period=30,
    entry_threshold=1.5,  # Standard deviations from mean
    exit_threshold=0.5
)

# Update vol statistics
vol_stats = strategy.update_volatility_statistics(implied_vol=0.20)

# Generate signal
signal = strategy.generate_signal(implied_vol=0.20, vol_stats=vol_stats)
```

### 3. Calendar Spreads

Profits from time decay and volatility term structure differences.

**Key Features:**
- Analysis of volatility term structure
- Short-term theta capture
- Long-term vega exposure
- Position rolling capabilities

**Use Case:** Sell front-month option (decay), buy back-month (vol exposure).

```python
from vol_arb.strategies.calendar_spread import CalendarSpreadStrategy

strategy = CalendarSpreadStrategy()

# Analyze spread opportunity
analysis = strategy.analyze_spread_opportunity(
    S=450, K=450,
    T_short=30/365, T_long=60/365,
    r=0.05,
    sigma_short=0.18, sigma_long=0.22,
    option_type='call'
)

if analysis['recommendation'] == 'ENTER':
    strategy.enter_spread(...)
```

### 4. Straddles and Strangles

Directional-neutral strategies for trading volatility and large price movements.

**Key Features:**
- Straddle: Buy ATM call + put
- Strangle: Buy OTM call + OTM put
- Breakeven analysis
- Scenario testing

**Use Case:** Profit from volatility increases or large price moves in either direction.

```python
from vol_arb.strategies.straddle_strangle import StraddleStrangleStrategy

strategy = StraddleStrangleStrategy()

# Enter long straddle
strategy.enter_long_straddle(
    S=450, K=450, T=30/365, r=0.05, sigma=0.20,
    quantity=10, call_price=5.0, put_price=5.0
)

# Enter long strangle (cheaper, requires larger move)
strategy.enter_long_strangle(
    S=450, K_call=455, K_put=445, T=30/365, r=0.05, sigma=0.20,
    quantity=10, call_price=3.0, put_price=3.0
)
```

## Volatility Calculations

The framework includes multiple volatility estimators:

### Historical Volatility Methods

```python
from vol_arb.utils.volatility import (
    historical_volatility,      # Close-to-close
    parkinson_volatility,       # High-low range
    garman_klass_volatility,    # OHLC
    yang_zhang_volatility,      # Handles overnight jumps
    ewma_volatility            # Exponentially weighted
)

# Example: Calculate historical volatility
hv = historical_volatility(prices, window=30)
```

### Volatility Cone

Analyze historical volatility ranges across multiple time horizons:

```python
from vol_arb.utils.volatility import volatility_cone

cone = volatility_cone(prices, windows=[10, 20, 30, 60, 90])
# Shows min, max, mean, percentiles for each window
```

## Option Pricing

Black-Scholes pricing and Greeks calculations:

```python
from vol_arb.utils.black_scholes import (
    black_scholes_price,
    calculate_greeks,
    implied_volatility
)

# Price an option
price = black_scholes_price(S=450, K=450, T=30/365, r=0.05, sigma=0.20, option_type='call')

# Calculate Greeks
greeks = calculate_greeks(S=450, K=450, T=30/365, r=0.05, sigma=0.20, option_type='call')
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.6f}")
print(f"Vega: ${greeks['vega']:.2f}")
print(f"Theta: ${greeks['theta']:.2f}/day")

# Calculate implied volatility
iv = implied_volatility(option_price=10.0, S=450, K=450, T=30/365, r=0.05, option_type='call')
```

## Market Data

Fetch real market data using yfinance:

```python
from vol_arb.data.market_data import MarketDataFetcher

fetcher = MarketDataFetcher()

# Get stock data
stock_data = fetcher.get_stock_data('AAPL', period='1y')

# Get options chain
calls, puts = fetcher.get_options_chain('AAPL')

# Get volatility surface
vol_surface = fetcher.get_volatility_surface('AAPL', num_expirations=5)

# Compare IV vs HV
comparison = fetcher.compare_implied_vs_historical('AAPL', hv_window=30)
```

## Key Concepts

### Volatility Arbitrage

Volatility arbitrage exploits the difference between:
- **Implied Volatility (IV)**: Market's expectation of future volatility (from option prices)
- **Realized/Historical Volatility (RV/HV)**: Actual volatility of the underlying

**Basic Strategy:**
- When IV < HV: Buy options (long volatility)
- When IV > HV: Sell options (short volatility)
- Maintain delta-neutral positions to isolate volatility exposure

### Greeks

- **Delta (Δ)**: Rate of change of option price with respect to stock price
- **Gamma (Γ)**: Rate of change of delta with respect to stock price
- **Theta (Θ)**: Rate of change of option price with respect to time
- **Vega (ν)**: Rate of change of option price with respect to volatility
- **Rho (ρ)**: Rate of change of option price with respect to interest rate

### Delta-Neutral Trading

A position where the total delta is zero, meaning small moves in the underlying don't affect the position value. This isolates volatility exposure.

### Gamma Scalping

Rebalancing a delta-neutral position to profit from realized volatility. As the stock moves, the position's delta changes (gamma effect), requiring rebalancing that captures profits.

## Risk Considerations

- **Model Risk**: Black-Scholes assumes constant volatility, log-normal returns
- **Transaction Costs**: Frequent rebalancing can be expensive
- **Gap Risk**: Large overnight moves can cause losses
- **Volatility Risk**: Wrong direction on volatility bets
- **Liquidity Risk**: Wide bid-ask spreads in options
- **Assignment Risk**: Short options can be assigned early

## Educational Use

This framework is designed for educational and research purposes. Always:
- Paper trade before using real money
- Understand the risks of each strategy
- Consider transaction costs and slippage
- Test strategies thoroughly with historical data
- Use proper risk management

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

MIT License - see LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Trading options and volatility strategies involves substantial risk. Past performance does not guarantee future results. Always consult with a financial advisor before trading.

## References

- Hull, J. C. (2018). *Options, Futures, and Other Derivatives*
- Sinclair, E. (2013). *Volatility Trading*
- Gatheral, J. (2006). *The Volatility Surface*
- Natenberg, S. (2015). *Option Volatility and Pricing*
