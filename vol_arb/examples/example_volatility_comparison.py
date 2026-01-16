"""
Example: Compare implied volatility vs historical volatility.

This script demonstrates how to identify volatility arbitrage opportunities
by comparing implied volatility from options markets with historical volatility.
"""
import sys
sys.path.append('..')

from vol_arb.data.market_data import MarketDataFetcher
from vol_arb.utils.volatility import volatility_cone
import pandas as pd


def main():
    """Main example function."""
    # Initialize data fetcher
    fetcher = MarketDataFetcher()

    # Example tickers to analyze
    tickers = ['AAPL', 'MSFT', 'SPY', 'TSLA']

    print("=" * 80)
    print("VOLATILITY ARBITRAGE OPPORTUNITY SCANNER")
    print("=" * 80)
    print()

    results = []

    for ticker in tickers:
        try:
            print(f"\nAnalyzing {ticker}...")
            print("-" * 40)

            # Compare IV vs HV
            comparison = fetcher.compare_implied_vs_historical(ticker, hv_window=30)

            print(f"Current Price: ${comparison['current_price']:.2f}")
            print(f"Implied Volatility (IV): {comparison['implied_vol']*100:.2f}%")
            print(f"Historical Volatility (HV): {comparison['historical_vol']*100:.2f}%")
            print(f"Volatility Spread (IV - HV): {comparison['vol_spread']*100:.2f}%")
            print(f"Volatility Ratio (IV / HV): {comparison['vol_ratio']:.2f}")
            print(f"Signal: {comparison['signal']}")

            results.append(comparison)

        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")

    print("\n" + "=" * 80)
    print("SUMMARY OF OPPORTUNITIES")
    print("=" * 80)

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('vol_spread', ascending=True)

        print("\nBest opportunities to BUY volatility (IV < HV):")
        buy_ops = df[df['signal'] == 'BUY_VOLATILITY']
        if not buy_ops.empty:
            for _, row in buy_ops.iterrows():
                print(f"  {row['ticker']}: IV={row['implied_vol']*100:.2f}%, "
                      f"HV={row['historical_vol']*100:.2f}%, "
                      f"Spread={row['vol_spread']*100:.2f}%")
        else:
            print("  None found")

        print("\nBest opportunities to SELL volatility (IV > HV):")
        sell_ops = df[df['signal'] == 'SELL_VOLATILITY']
        if not sell_ops.empty:
            for _, row in sell_ops.iterrows():
                print(f"  {row['ticker']}: IV={row['implied_vol']*100:.2f}%, "
                      f"HV={row['historical_vol']*100:.2f}%, "
                      f"Spread={row['vol_spread']*100:.2f}%")
        else:
            print("  None found")

    print("\n" + "=" * 80)


def analyze_volatility_cone_example():
    """Example of volatility cone analysis."""
    fetcher = MarketDataFetcher()
    ticker = 'SPY'

    print(f"\n\nVOLATILITY CONE ANALYSIS FOR {ticker}")
    print("=" * 80)

    # Get historical data
    df = fetcher.get_stock_data(ticker, period='2y')

    # Calculate volatility cone
    cone = volatility_cone(df['Close'], windows=[10, 20, 30, 60, 90, 120])

    print("\nHistorical Volatility Statistics:")
    print(cone.to_string(index=False))

    # Get current IV
    try:
        iv_data = fetcher.get_implied_volatility_from_options(ticker)
        current_iv = iv_data['atm_iv'] * 100

        print(f"\n\nCurrent Implied Volatility: {current_iv:.2f}%")
        print("\nComparison to Historical Ranges:")

        for _, row in cone.iterrows():
            window = row['window']
            current_hv = row['current']
            mean_hv = row['mean']
            p25 = row['p25']
            p75 = row['p75']

            print(f"\n{window}-day window:")
            print(f"  Current HV: {current_hv*100:.2f}%")
            print(f"  Mean HV: {mean_hv*100:.2f}%")
            print(f"  25th-75th percentile: {p25*100:.2f}% - {p75*100:.2f}%")

            if current_iv > p75 * 100:
                print(f"  => IV is ABOVE 75th percentile (consider selling volatility)")
            elif current_iv < p25 * 100:
                print(f"  => IV is BELOW 25th percentile (consider buying volatility)")
            else:
                print(f"  => IV is within normal range")

    except Exception as e:
        print(f"\nCould not fetch IV data: {e}")


if __name__ == '__main__':
    main()
    analyze_volatility_cone_example()
