"""
Example: Long straddle strategy for volatility trading.

This script demonstrates how to use the straddle strategy to profit from
volatility increases or large price movements.
"""
import sys
sys.path.append('..')

from vol_arb.strategies.straddle_strangle import StraddleStrangleStrategy
from vol_arb.data.market_data import MarketDataFetcher
import numpy as np


def main():
    """Main example function."""
    print("=" * 80)
    print("STRADDLE STRATEGY EXAMPLE")
    print("=" * 80)
    print()

    # Initialize strategy
    strategy = StraddleStrangleStrategy(
        transaction_cost=0.001,
        profit_target_pct=0.50,
        stop_loss_pct=0.50
    )

    # Example parameters
    ticker = 'SPY'
    S = 450.0  # Current stock price
    K = 450.0  # Strike price (ATM)
    T = 30 / 365  # 30 days to expiration
    r = 0.05  # 5% risk-free rate
    sigma = 0.20  # 20% implied volatility

    print(f"Underlying: {ticker}")
    print(f"Stock Price: ${S:.2f}")
    print(f"Strike: ${K:.2f}")
    print(f"Time to Expiration: {T*365:.0f} days")
    print(f"Implied Volatility: {sigma*100:.1f}%")
    print()

    # Calculate straddle metrics
    print("ANALYZING STRADDLE OPPORTUNITY")
    print("-" * 80)

    metrics = strategy.calculate_straddle_metrics(S, K, T, r, sigma)

    print(f"Call Price: ${metrics['call_price']:.2f}")
    print(f"Put Price: ${metrics['put_price']:.2f}")
    print(f"Total Cost: ${metrics['total_cost']:.2f}")
    print()

    print("Greeks:")
    print(f"  Net Delta: {metrics['net_delta']:.4f} (should be near 0 for ATM)")
    print(f"  Net Gamma: {metrics['net_gamma']:.6f}")
    print(f"  Net Theta: ${metrics['net_theta']:.2f} per day")
    print(f"  Net Vega: ${metrics['net_vega']:.2f} per 1% vol change")
    print()

    print("Breakeven Analysis:")
    print(f"  Upper Breakeven: ${metrics['upper_breakeven']:.2f} "
          f"({(metrics['upper_breakeven']/S - 1)*100:.1f}% move)")
    print(f"  Lower Breakeven: ${metrics['lower_breakeven']:.2f} "
          f"({(metrics['lower_breakeven']/S - 1)*100:.1f}% move)")
    print(f"  Required Move: ±{metrics['breakeven_move_pct']:.1f}%")
    print()

    # Enter the straddle
    print("ENTERING LONG STRADDLE")
    print("-" * 80)

    quantity = 10  # 10 straddles
    trade = strategy.enter_long_straddle(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        quantity=quantity,
        call_price=metrics['call_price'],
        put_price=metrics['put_price']
    )

    print(f"Entered {quantity} long straddles")
    print(f"Total Cost: ${trade['total_cost']:.2f}")
    print(f"Net Vega: ${trade['net_vega']:.2f}")
    print()

    # Simulate different scenarios
    print("SCENARIO ANALYSIS")
    print("=" * 80)
    print()

    scenarios = [
        {"name": "Large Up Move", "S_new": 470, "sigma_new": 0.25, "T_new": 20/365},
        {"name": "Large Down Move", "S_new": 430, "sigma_new": 0.30, "T_new": 20/365},
        {"name": "Small Move, Vol Up", "S_new": 452, "sigma_new": 0.30, "T_new": 20/365},
        {"name": "No Move, Vol Down", "S_new": 450, "sigma_new": 0.15, "T_new": 20/365},
        {"name": "No Move, Time Decay", "S_new": 450, "sigma_new": 0.20, "T_new": 10/365},
    ]

    for scenario in scenarios:
        print(f"Scenario: {scenario['name']}")
        print("-" * 40)

        S_new = scenario['S_new']
        sigma_new = scenario['sigma_new']
        T_new = scenario['T_new']

        # Mark to market
        mtm = strategy.mark_to_market(0, S_new, T_new, sigma_new)

        print(f"  Stock Price: ${S_new:.2f} ({(S_new/S-1)*100:.1f}% change)")
        print(f"  Implied Vol: {sigma_new*100:.1f}% ({(sigma_new-sigma)*100:.1f}% change)")
        print(f"  Days Remaining: {T_new*365:.0f}")
        print(f"  Position Value: ${mtm['position_value']:.2f}")
        print(f"  P&L: ${mtm['unrealized_pnl']:.2f} ({mtm['pnl_pct']:.1f}%)")

        if mtm['pnl_pct'] > 0:
            print(f"  Status: PROFITABLE ✓")
        else:
            print(f"  Status: LOSS ✗")

        # Check exit conditions
        should_exit, reason = strategy.check_exit_conditions(0, S_new, T_new, sigma_new)
        if should_exit:
            print(f"  Exit Signal: {reason}")

        print()

    # Show performance summary
    print("STRATEGY PERFORMANCE")
    print("=" * 80)
    summary = strategy.get_performance_summary()
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Open Positions: {len(strategy.positions)}")
    print()

    print("Trade History:")
    if len(strategy.trades) > 0:
        print(strategy.get_trade_history().to_string(index=False))


def compare_straddle_vs_strangle():
    """Compare straddle vs strangle strategies."""
    print("\n\n" + "=" * 80)
    print("STRADDLE VS STRANGLE COMPARISON")
    print("=" * 80)
    print()

    strategy = StraddleStrangleStrategy()

    S = 450.0
    T = 30 / 365
    r = 0.05
    sigma = 0.20

    # Straddle (ATM)
    K_straddle = 450.0
    straddle_metrics = strategy.calculate_straddle_metrics(S, K_straddle, T, r, sigma)

    # Strangle (OTM)
    K_call = 455.0  # 5 points OTM
    K_put = 445.0   # 5 points OTM
    strangle_metrics = strategy.calculate_strangle_metrics(S, K_call, K_put, T, r, sigma)

    print("STRADDLE (ATM - Strike $450)")
    print("-" * 40)
    print(f"Total Cost: ${straddle_metrics['total_cost']:.2f}")
    print(f"Breakeven Move: ±{straddle_metrics['breakeven_move_pct']:.1f}%")
    print(f"Net Vega: ${straddle_metrics['net_vega']:.2f}")
    print(f"Net Theta: ${straddle_metrics['net_theta']:.2f}/day")
    print()

    print("STRANGLE (OTM - Strikes $445/$455)")
    print("-" * 40)
    print(f"Total Cost: ${strangle_metrics['total_cost']:.2f}")
    print(f"Breakeven Move Up: +{strangle_metrics['breakeven_move_pct_up']:.1f}%")
    print(f"Breakeven Move Down: -{strangle_metrics['breakeven_move_pct_down']:.1f}%")
    print(f"Net Vega: ${strangle_metrics['net_vega']:.2f}")
    print(f"Net Theta: ${strangle_metrics['net_theta']:.2f}/day")
    print()

    cost_savings = straddle_metrics['total_cost'] - strangle_metrics['total_cost']
    print("COMPARISON")
    print("-" * 40)
    print(f"Cost Savings (Strangle): ${cost_savings:.2f} "
          f"({cost_savings/straddle_metrics['total_cost']*100:.1f}%)")
    print(f"Vega Difference: ${(straddle_metrics['net_vega'] - strangle_metrics['net_vega']):.2f}")
    print()

    print("Trade-offs:")
    print("  Straddle:")
    print("    + Higher vega exposure (more sensitive to vol changes)")
    print("    + Lower breakeven moves required")
    print("    - More expensive")
    print("    - Higher theta decay")
    print()
    print("  Strangle:")
    print("    + Cheaper to enter")
    print("    + Lower theta decay")
    print("    - Requires larger moves to profit")
    print("    - Lower vega exposure")


if __name__ == '__main__':
    main()
    compare_straddle_vs_strangle()
