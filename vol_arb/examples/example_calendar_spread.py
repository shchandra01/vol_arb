"""
Example: Calendar spread strategy for volatility term structure trading.

This script demonstrates how to use calendar spreads to profit from
time decay and volatility term structure.
"""
import sys
sys.path.append('..')

from vol_arb.strategies.calendar_spread import CalendarSpreadStrategy
import numpy as np


def main():
    """Main example function."""
    print("=" * 80)
    print("CALENDAR SPREAD STRATEGY EXAMPLE")
    print("=" * 80)
    print()

    # Initialize strategy
    strategy = CalendarSpreadStrategy(
        transaction_cost=0.001,
        max_loss_pct=0.50
    )

    # Example parameters
    ticker = 'SPY'
    S = 450.0  # Current stock price
    K = 450.0  # Strike price (same for both legs)
    T_short = 30 / 365  # 30 days to expiration (front month)
    T_long = 60 / 365   # 60 days to expiration (back month)
    r = 0.05  # 5% risk-free rate
    sigma_short = 0.18  # 18% IV for front month
    sigma_long = 0.22   # 22% IV for back month (upward sloping term structure)
    option_type = 'call'

    print(f"Underlying: {ticker}")
    print(f"Stock Price: ${S:.2f}")
    print(f"Strike: ${K:.2f} (ATM)")
    print(f"Option Type: {option_type.upper()}")
    print()
    print(f"Front Month (SHORT leg):")
    print(f"  Time to Expiration: {T_short*365:.0f} days")
    print(f"  Implied Volatility: {sigma_short*100:.1f}%")
    print()
    print(f"Back Month (LONG leg):")
    print(f"  Time to Expiration: {T_long*365:.0f} days")
    print(f"  Implied Volatility: {sigma_long*100:.1f}%")
    print()

    # Analyze the spread opportunity
    print("ANALYZING CALENDAR SPREAD OPPORTUNITY")
    print("-" * 80)

    analysis = strategy.analyze_spread_opportunity(
        S, K, T_short, T_long, r, sigma_short, sigma_long, option_type
    )

    print(f"Front Month Price: ${analysis['price_short']:.2f}")
    print(f"Back Month Price: ${analysis['price_long']:.2f}")
    print(f"Net Debit: ${analysis['net_debit']:.2f}")
    print()

    print("Volatility Term Structure:")
    print(f"  Vol Spread (Long - Short): {analysis['vol_spread']*100:.2f}%")
    print(f"  Vol Ratio (Long / Short): {analysis['vol_ratio']:.2f}")
    print()

    print("Position Greeks:")
    print(f"  Net Delta: {analysis['net_delta']:.4f}")
    print(f"  Net Gamma: {analysis['net_gamma']:.6f}")
    print(f"  Net Theta: ${analysis['net_theta']:.2f} per day")
    print(f"  Net Vega: ${analysis['net_vega']:.2f} per 1% vol change")
    print()

    print(f"Opportunity Score: {analysis['opportunity_score']:.2f}")
    print(f"Recommendation: {analysis['recommendation']}")
    print()

    # Enter the spread
    if analysis['recommendation'] == 'ENTER':
        print("ENTERING CALENDAR SPREAD")
        print("-" * 80)

        quantity = 10  # 10 spreads
        trade = strategy.enter_spread(
            S=S,
            K=K,
            T_short=T_short,
            T_long=T_long,
            r=r,
            sigma_short=sigma_short,
            sigma_long=sigma_long,
            option_type=option_type,
            quantity=quantity,
            price_short=analysis['price_short'],
            price_long=analysis['price_long']
        )

        print(f"Entered {quantity} calendar spreads")
        print(f"Short {quantity} x {T_short*365:.0f}-day ${K} {option_type}s @ ${trade['price_short']:.2f}")
        print(f"Long {quantity} x {T_long*365:.0f}-day ${K} {option_type}s @ ${trade['price_long']:.2f}")
        print(f"Net Debit: ${trade['net_debit']:.2f}")
        print()

        # Simulate time passage and price scenarios
        print("SCENARIO ANALYSIS")
        print("=" * 80)
        print()

        # Scenario 1: Time decay, stock stays near strike
        print("Scenario 1: TIME DECAY - Stock stays near strike")
        print("-" * 40)

        days_passed = 10
        T_short_new = (T_short * 365 - days_passed) / 365
        T_long_new = (T_long * 365 - days_passed) / 365
        S_new = 450.0
        sigma_short_new = 0.18
        sigma_long_new = 0.22

        mtm = strategy.mark_to_market(
            0, S_new, T_short_new, T_long_new, sigma_short_new, sigma_long_new
        )

        print(f"  Days Passed: {days_passed}")
        print(f"  Stock Price: ${S_new:.2f} (unchanged)")
        print(f"  Short Leg: {T_short_new*365:.0f} days remaining")
        print(f"  Long Leg: {T_long_new*365:.0f} days remaining")
        print(f"  Position Value: ${mtm['net_value']:.2f}")
        print(f"  P&L: ${mtm['unrealized_pnl']:.2f} ({mtm['pnl_pct']:.1f}%)")
        print()

        # Scenario 2: Volatility increase
        print("Scenario 2: VOLATILITY SPIKE")
        print("-" * 40)

        T_short_new = (T_short * 365 - 5) / 365
        T_long_new = (T_long * 365 - 5) / 365
        S_new = 450.0
        sigma_short_new = 0.25  # Vol increased
        sigma_long_new = 0.30   # Vol increased more in back month

        mtm = strategy.mark_to_market(
            0, S_new, T_short_new, T_long_new, sigma_short_new, sigma_long_new
        )

        print(f"  Days Passed: 5")
        print(f"  Stock Price: ${S_new:.2f} (unchanged)")
        print(f"  Vol Change: Short +{(sigma_short_new-sigma_short)*100:.1f}%, "
              f"Long +{(sigma_long_new-sigma_long)*100:.1f}%")
        print(f"  Position Value: ${mtm['net_value']:.2f}")
        print(f"  P&L: ${mtm['unrealized_pnl']:.2f} ({mtm['pnl_pct']:.1f}%)")
        print()

        # Scenario 3: Large price move (risk scenario)
        print("Scenario 3: LARGE PRICE MOVE (Risk)")
        print("-" * 40)

        T_short_new = (T_short * 365 - 10) / 365
        T_long_new = (T_long * 365 - 10) / 365
        S_new = 465.0  # Large move up
        sigma_short_new = 0.20
        sigma_long_new = 0.24

        mtm = strategy.mark_to_market(
            0, S_new, T_short_new, T_long_new, sigma_short_new, sigma_long_new
        )

        print(f"  Days Passed: 10")
        print(f"  Stock Price: ${S_new:.2f} (+{(S_new/S-1)*100:.1f}%)")
        print(f"  Position Value: ${mtm['net_value']:.2f}")
        print(f"  P&L: ${mtm['unrealized_pnl']:.2f} ({mtm['pnl_pct']:.1f}%)")

        # Check exit conditions
        should_exit, reason = strategy.check_exit_conditions(
            0, S_new, T_short_new, T_long_new, sigma_short_new, sigma_long_new
        )
        if should_exit:
            print(f"  Exit Signal: {reason}")
        print()

        # Scenario 4: Near expiration of short leg
        print("Scenario 4: SHORT LEG NEAR EXPIRATION")
        print("-" * 40)

        T_short_new = 5 / 365  # 5 days left
        T_long_new = 35 / 365  # 35 days left
        S_new = 452.0
        sigma_short_new = 0.18
        sigma_long_new = 0.22

        mtm = strategy.mark_to_market(
            0, S_new, T_short_new, T_long_new, sigma_short_new, sigma_long_new
        )

        print(f"  Short Leg: {T_short_new*365:.0f} days remaining")
        print(f"  Long Leg: {T_long_new*365:.0f} days remaining")
        print(f"  Stock Price: ${S_new:.2f}")
        print(f"  Position Value: ${mtm['net_value']:.2f}")
        print(f"  P&L: ${mtm['unrealized_pnl']:.2f} ({mtm['pnl_pct']:.1f}%)")

        should_exit, reason = strategy.check_exit_conditions(
            0, S_new, T_short_new, T_long_new, sigma_short_new, sigma_long_new
        )
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

        print("Key Insights for Calendar Spreads:")
        print("-" * 40)
        print("✓ Profit from time decay when stock stays near strike")
        print("✓ Benefit from volatility increases (long vega)")
        print("✓ Work best with upward-sloping vol term structure")
        print("✗ Risk: Large price moves can cause losses")
        print("✗ Need to manage/roll the short leg before expiration")


if __name__ == '__main__':
    main()
