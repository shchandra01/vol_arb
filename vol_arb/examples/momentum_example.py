"""
Example usage of Cross-Sectional Momentum Signal.

This example demonstrates:
1. Loading price data for multiple stocks
2. Configuring the momentum signal
3. Generating signals and constructing portfolios
4. Running a backtest
5. Analyzing results
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the momentum signal
from vol_arb.signals.momentum import (
    CrossSectionalMomentum,
    MomentumConfig,
    MomentumType
)
from vol_arb.data.market_data import MarketDataFetcher


def run_momentum_example():
    """Run a complete momentum strategy example."""

    print("=" * 60)
    print("Cross-Sectional Momentum Strategy Example")
    print("=" * 60)

    # 1. Setup: Define universe and fetch data
    # ----------------------------------------
    print("\n1. Fetching price data...")

    # Example universe (tech + financials for demonstration)
    universe = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech
        'JPM', 'BAC', 'GS', 'MS', 'C',             # Financials
        'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV',        # Healthcare
        'XOM', 'CVX', 'COP', 'SLB', 'EOG',         # Energy
    ]

    # Sector mapping for neutralization
    sectors = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'AMZN': 'Technology', 'META': 'Technology',
        'JPM': 'Financials', 'BAC': 'Financials', 'GS': 'Financials',
        'MS': 'Financials', 'C': 'Financials',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
        'MRK': 'Healthcare', 'ABBV': 'Healthcare',
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        'SLB': 'Energy', 'EOG': 'Energy',
    }

    fetcher = MarketDataFetcher()

    # Fetch data for all stocks
    prices_dict = {}
    for ticker in universe:
        try:
            df = fetcher.get_stock_data(ticker, period='2y')
            prices_dict[ticker] = df['Close']
            print(f"  Loaded {ticker}: {len(df)} days")
        except Exception as e:
            print(f"  Failed to load {ticker}: {e}")

    # Combine into DataFrame
    prices = pd.DataFrame(prices_dict)
    print(f"\nTotal: {len(prices)} days, {len(prices.columns)} stocks")

    # 2. Configure the momentum signal
    # --------------------------------
    print("\n2. Configuring momentum signal...")

    # Classic 12-1 momentum configuration
    config = MomentumConfig(
        lookback_start=252,      # 12 months lookback
        lookback_end=21,         # Skip most recent month
        momentum_type=MomentumType.CUMULATIVE,
        long_percentile=0.2,     # Top 20%
        short_percentile=0.2,    # Bottom 20%
        volatility_scale=True,   # Risk-adjust signals
        vol_lookback=60,         # 60-day vol for scaling
        winsorize=True,          # Handle outliers
        winsorize_pct=0.01,      # 1% / 99% winsorization
        sector_neutralize=True,  # Neutralize sector bets
        max_position_weight=0.1, # Max 10% per stock
        min_stocks_per_side=3,   # Minimum 3 stocks long/short
    )

    print(f"  Lookback: {config.lookback_start} days")
    print(f"  Skip recent: {config.lookback_end} days")
    print(f"  Volatility scaling: {config.volatility_scale}")
    print(f"  Sector neutralization: {config.sector_neutralize}")
    print(f"  Long/Short: top/bottom {config.long_percentile*100:.0f}%")

    # 3. Initialize and generate signals
    # ----------------------------------
    print("\n3. Generating momentum signals...")

    momentum = CrossSectionalMomentum(config)
    momentum.update_prices(prices, sectors)

    # Generate signals for latest date
    signals = momentum.generate_signals()

    print(f"\n  Generated signals for {len(signals)} stocks:")
    signal_df = pd.DataFrame.from_dict(signals, orient='index', columns=['momentum'])
    signal_df = signal_df.sort_values('momentum', ascending=False)

    print("\n  Top 5 (highest momentum):")
    for ticker, row in signal_df.head().iterrows():
        print(f"    {ticker}: {row['momentum']:.4f}")

    print("\n  Bottom 5 (lowest momentum):")
    for ticker, row in signal_df.tail().iterrows():
        print(f"    {ticker}: {row['momentum']:.4f}")

    # Signal statistics
    stats = momentum.get_signal_statistics(signals)
    print(f"\n  Signal Statistics:")
    print(f"    Mean: {stats['mean']:.4f}")
    print(f"    Std: {stats['std']:.4f}")
    print(f"    % Positive: {stats['pct_positive']*100:.1f}%")

    # 4. Construct portfolio
    # ----------------------
    print("\n4. Constructing portfolio...")

    portfolio = momentum.construct_portfolio(signals)

    print(f"\n  Long positions ({portfolio.long_count} stocks):")
    for ticker, weight in sorted(portfolio.long_weights.items(), key=lambda x: -x[1]):
        print(f"    {ticker}: {weight*100:.1f}%")

    print(f"\n  Short positions ({portfolio.short_count} stocks):")
    for ticker, weight in sorted(portfolio.short_weights.items(), key=lambda x: x[1]):
        print(f"    {ticker}: {weight*100:.1f}%")

    print(f"\n  Net exposure: {portfolio.net_exposure:.4f}")
    print(f"  Gross exposure: {portfolio.gross_exposure:.4f}")

    # Sector exposures
    exposures = momentum.get_current_exposures(portfolio)
    print(f"\n  Sector Exposures:")
    for sector, exp in exposures.items():
        print(f"    {sector}: Net={exp['net']*100:+.1f}%")

    # 5. Run backtest
    # ---------------
    print("\n5. Running backtest...")

    backtest_results = momentum.backtest_signal(
        prices=prices,
        rebalance_frequency=21,  # Monthly rebalance
        transaction_cost=0.001,  # 10 bps
        sectors=sectors
    )

    print(f"\n  Backtest period: {backtest_results.index[0].date()} to {backtest_results.index[-1].date()}")
    print(f"  Total days: {len(backtest_results)}")

    # Performance metrics
    metrics = momentum.get_performance_metrics(backtest_results)

    print(f"\n  Performance Metrics:")
    print(f"    Total Return: {metrics['total_return']*100:.1f}%")
    print(f"    Annualized Return: {metrics['annualized_return']*100:.1f}%")
    print(f"    Annualized Volatility: {metrics['annualized_volatility']*100:.1f}%")
    print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"    Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"    Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
    print(f"    Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"    Avg Annual Turnover: {metrics['avg_annual_turnover']*100:.1f}%")

    # 6. Signal decay analysis
    # ------------------------
    print("\n6. Analyzing signal decay...")

    try:
        decay = momentum.get_signal_decay_analysis(
            prices=prices,
            horizons=[1, 5, 10, 21, 63],  # 1d, 1w, 2w, 1m, 3m
            sectors=sectors
        )

        print("\n  Information Coefficient (IC) by Horizon:")
        for _, row in decay.iterrows():
            print(f"    {row['horizon']:2d} days: IC={row['mean_ic']:.3f}, "
                  f"IC_IR={row['ic_ir']:.2f}, "
                  f"%Positive={row['pct_positive']*100:.1f}%")
    except Exception as e:
        print(f"  Signal decay analysis requires scipy: {e}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)

    return momentum, backtest_results, metrics


def run_parameter_sensitivity():
    """Demonstrate parameter sensitivity analysis."""

    print("\n" + "=" * 60)
    print("Parameter Sensitivity Analysis")
    print("=" * 60)

    # Fetch a small universe for quick testing
    universe = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC', 'JNJ', 'XOM', 'CVX']

    fetcher = MarketDataFetcher()
    prices_dict = {}
    for ticker in universe:
        try:
            df = fetcher.get_stock_data(ticker, period='2y')
            prices_dict[ticker] = df['Close']
        except:
            pass

    prices = pd.DataFrame(prices_dict)

    # Test different lookback periods
    lookbacks = [126, 189, 252]  # 6mo, 9mo, 12mo
    skip_periods = [0, 21, 42]   # 0, 1mo, 2mo skip

    print("\nLookback and Skip Period Sensitivity:")
    print("-" * 50)

    for lookback in lookbacks:
        for skip in skip_periods:
            config = MomentumConfig(
                lookback_start=lookback,
                lookback_end=skip,
                volatility_scale=True,
                long_percentile=0.25,
                short_percentile=0.25,
            )

            mom = CrossSectionalMomentum(config)

            try:
                results = mom.backtest_signal(prices, rebalance_frequency=21)
                metrics = mom.get_performance_metrics(results)

                print(f"  Lookback={lookback:3d}, Skip={skip:2d}: "
                      f"Sharpe={metrics['sharpe_ratio']:+.2f}, "
                      f"Return={metrics['annualized_return']*100:+.1f}%")
            except Exception as e:
                print(f"  Lookback={lookback:3d}, Skip={skip:2d}: Error - {e}")


if __name__ == '__main__':
    # Run main example
    momentum, results, metrics = run_momentum_example()

    # Optional: parameter sensitivity
    # run_parameter_sensitivity()
