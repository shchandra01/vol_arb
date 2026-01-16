"""
Calendar spread (time spread) strategy.

A calendar spread involves buying and selling options with the same strike
but different expiration dates. This strategy profits from differences in
implied volatility across expiration dates and time decay.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ..utils.black_scholes import black_scholes_price, calculate_greeks


class CalendarSpreadStrategy:
    """
    Calendar spread strategy for volatility arbitrage.

    The strategy:
    1. Sell near-term option (front month) - collects premium, high theta
    2. Buy far-term option (back month) - long vega exposure
    3. Profits when:
       - Near-term option decays faster (time decay)
       - Volatility increases (benefits long back month)
       - Volatility term structure is upward sloping
    4. Risk: Large moves in underlying can cause losses
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,
        max_loss_pct: float = 0.50
    ):
        """
        Initialize calendar spread strategy.

        Args:
            transaction_cost: Transaction cost as fraction of trade value
            max_loss_pct: Maximum loss as percentage of initial credit/debit
        """
        self.transaction_cost = transaction_cost
        self.max_loss_pct = max_loss_pct
        self.positions = []
        self.trades = []

    def analyze_spread_opportunity(
        self,
        S: float,
        K: float,
        T_short: float,
        T_long: float,
        r: float,
        sigma_short: float,
        sigma_long: float,
        option_type: str = 'call'
    ) -> Dict:
        """
        Analyze calendar spread opportunity.

        Args:
            S: Current stock price
            K: Strike price (same for both legs)
            T_short: Time to expiration of short option (years)
            T_long: Time to expiration of long option (years)
            r: Risk-free rate
            sigma_short: Implied vol of short option
            sigma_long: Implied vol of long option
            option_type: 'call' or 'put'

        Returns:
            Dictionary with spread analysis
        """
        if T_long <= T_short:
            raise ValueError("Long option must have longer expiration than short option")

        # Calculate prices
        price_short = black_scholes_price(S, K, T_short, r, sigma_short, option_type)
        price_long = black_scholes_price(S, K, T_long, r, sigma_long, option_type)

        # Net debit (we pay to enter the spread)
        net_debit = price_long - price_short

        # Calculate Greeks
        greeks_short = calculate_greeks(S, K, T_short, r, sigma_short, option_type)
        greeks_long = calculate_greeks(S, K, T_long, r, sigma_long, option_type)

        # Net position Greeks (long back month - short front month)
        net_delta = greeks_long['delta'] - greeks_short['delta']
        net_gamma = greeks_long['gamma'] - greeks_short['gamma']
        net_theta = greeks_long['theta'] - greeks_short['theta']
        net_vega = greeks_long['vega'] - greeks_short['vega']

        # Volatility term structure
        vol_spread = sigma_long - sigma_short
        vol_ratio = sigma_long / sigma_short if sigma_short > 0 else np.inf

        # Opportunity score (higher is better)
        # Favor: positive vol spread, positive net vega, negative net theta
        opportunity_score = vol_spread * 10 + net_vega - abs(net_theta)

        return {
            'net_debit': net_debit,
            'price_short': price_short,
            'price_long': price_long,
            'vol_spread': vol_spread,
            'vol_ratio': vol_ratio,
            'net_delta': net_delta,
            'net_gamma': net_gamma,
            'net_theta': net_theta,
            'net_vega': net_vega,
            'greeks_short': greeks_short,
            'greeks_long': greeks_long,
            'opportunity_score': opportunity_score,
            'recommendation': 'ENTER' if (vol_spread > 0.02 and net_vega > 0) else 'PASS'
        }

    def enter_spread(
        self,
        S: float,
        K: float,
        T_short: float,
        T_long: float,
        r: float,
        sigma_short: float,
        sigma_long: float,
        option_type: str,
        quantity: int,
        price_short: float,
        price_long: float
    ) -> Dict:
        """
        Enter a calendar spread position.

        Args:
            S: Current stock price
            K: Strike price
            T_short: Time to expiration of short option
            T_long: Time to expiration of long option
            r: Risk-free rate
            sigma_short: Implied vol of short option
            sigma_long: Implied vol of long option
            option_type: 'call' or 'put'
            quantity: Number of spreads (each spread is 1 short + 1 long)
            price_short: Market price of short option
            price_long: Market price of long option

        Returns:
            Trade details
        """
        # Calculate net cost
        cost_long = price_long * quantity * 100
        proceeds_short = price_short * quantity * 100
        net_debit = cost_long - proceeds_short

        transaction_cost = (cost_long + proceeds_short) * self.transaction_cost
        total_cost = net_debit + transaction_cost

        position = {
            'S': S,
            'K': K,
            'T_short': T_short,
            'T_long': T_long,
            'r': r,
            'sigma_short': sigma_short,
            'sigma_long': sigma_long,
            'option_type': option_type,
            'quantity': quantity,
            'entry_price_short': price_short,
            'entry_price_long': price_long,
            'entry_stock_price': S,
            'net_debit': total_cost,
            'entry_date': None  # Would be set with actual date
        }

        self.positions.append(position)

        trade = {
            'action': 'ENTER_SPREAD',
            'stock_price': S,
            'strike': K,
            'option_type': option_type,
            'quantity': quantity,
            'price_short': price_short,
            'price_long': price_long,
            'net_debit': total_cost,
            'vol_short': sigma_short,
            'vol_long': sigma_long,
            'vol_spread': sigma_long - sigma_short
        }

        self.trades.append(trade)
        return trade

    def mark_to_market(
        self,
        position_index: int,
        S: float,
        T_short: float,
        T_long: float,
        sigma_short: float,
        sigma_long: float
    ) -> Dict:
        """
        Mark a spread position to market.

        Args:
            position_index: Index of position to mark
            S: Current stock price
            T_short: Updated time to expiration of short option
            T_long: Updated time to expiration of long option
            sigma_short: Current implied vol of short option
            sigma_long: Current implied vol of long option

        Returns:
            Current position value and P&L
        """
        if position_index >= len(self.positions):
            raise ValueError(f"Invalid position index: {position_index}")

        pos = self.positions[position_index]

        # Current prices
        current_price_short = black_scholes_price(
            S, pos['K'], T_short, pos['r'], sigma_short, pos['option_type']
        )
        current_price_long = black_scholes_price(
            S, pos['K'], T_long, pos['r'], sigma_long, pos['option_type']
        )

        # Position values
        value_short = current_price_short * pos['quantity'] * 100  # We're short
        value_long = current_price_long * pos['quantity'] * 100  # We're long

        # Net position value
        net_value = value_long - value_short

        # P&L
        unrealized_pnl = net_value - pos['net_debit']

        # Greeks
        greeks_short = calculate_greeks(
            S, pos['K'], T_short, pos['r'], sigma_short, pos['option_type']
        )
        greeks_long = calculate_greeks(
            S, pos['K'], T_long, pos['r'], sigma_long, pos['option_type']
        )

        return {
            'position_index': position_index,
            'stock_price': S,
            'value_short': value_short,
            'value_long': value_long,
            'net_value': net_value,
            'entry_cost': pos['net_debit'],
            'unrealized_pnl': unrealized_pnl,
            'pnl_pct': (unrealized_pnl / pos['net_debit'] * 100) if pos['net_debit'] > 0 else 0,
            'net_delta': greeks_long['delta'] - greeks_short['delta'],
            'net_vega': greeks_long['vega'] - greeks_short['vega'],
            'net_theta': greeks_long['theta'] - greeks_short['theta'],
            'vol_spread': sigma_long - sigma_short
        }

    def check_exit_conditions(
        self,
        position_index: int,
        S: float,
        T_short: float,
        T_long: float,
        sigma_short: float,
        sigma_long: float
    ) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Args:
            position_index: Index of position to check
            S: Current stock price
            T_short: Current time to expiration of short option
            T_long: Current time to expiration of long option
            sigma_short: Current implied vol of short option
            sigma_long: Current implied vol of long option

        Returns:
            Tuple of (should_exit, reason)
        """
        pos = self.positions[position_index]
        mtm = self.mark_to_market(
            position_index, S, T_short, T_long, sigma_short, sigma_long
        )

        # Exit condition 1: Max loss reached
        if mtm['pnl_pct'] < -self.max_loss_pct * 100:
            return True, 'MAX_LOSS'

        # Exit condition 2: Short option near expiration
        if T_short < 7/365:  # Less than 7 days
            return True, 'SHORT_EXPIRING'

        # Exit condition 3: Target profit reached (100% of debit)
        if mtm['pnl_pct'] > 100:
            return True, 'TARGET_PROFIT'

        # Exit condition 4: Volatility term structure inverted significantly
        if sigma_long < sigma_short - 0.05:
            return True, 'VOL_INVERSION'

        return False, 'HOLD'

    def close_spread(
        self,
        position_index: int,
        S: float,
        price_short: float,
        price_long: float,
        reason: str = 'MANUAL'
    ) -> Dict:
        """
        Close a calendar spread position.

        Args:
            position_index: Index of position to close
            S: Current stock price
            price_short: Current market price of short option
            price_long: Current market price of long option
            reason: Reason for closing

        Returns:
            Trade details
        """
        if position_index >= len(self.positions):
            raise ValueError(f"Invalid position index: {position_index}")

        pos = self.positions[position_index]

        # Calculate proceeds/costs
        cost_short = price_short * pos['quantity'] * 100  # Buying back short
        proceeds_long = price_long * pos['quantity'] * 100  # Selling long

        net_proceeds = proceeds_long - cost_short
        transaction_cost = (cost_short + proceeds_long) * self.transaction_cost
        net_proceeds_after_cost = net_proceeds - transaction_cost

        # Total P&L
        total_pnl = net_proceeds_after_cost - pos['net_debit']
        pnl_pct = (total_pnl / pos['net_debit'] * 100) if pos['net_debit'] > 0 else 0

        trade = {
            'action': 'CLOSE_SPREAD',
            'reason': reason,
            'stock_price': S,
            'strike': pos['K'],
            'option_type': pos['option_type'],
            'quantity': pos['quantity'],
            'entry_price_short': pos['entry_price_short'],
            'entry_price_long': pos['entry_price_long'],
            'exit_price_short': price_short,
            'exit_price_long': price_long,
            'entry_stock_price': pos['entry_stock_price'],
            'exit_stock_price': S,
            'stock_move': S - pos['entry_stock_price'],
            'entry_net_debit': pos['net_debit'],
            'exit_net_proceeds': net_proceeds_after_cost,
            'pnl': total_pnl,
            'pnl_pct': pnl_pct
        }

        self.trades.append(trade)
        self.positions.pop(position_index)

        return trade

    def roll_short_leg(
        self,
        position_index: int,
        S: float,
        new_T_short: float,
        new_sigma_short: float,
        close_price_short: float,
        new_price_short: float
    ) -> Dict:
        """
        Roll the short leg to a further expiration (common management technique).

        Args:
            position_index: Index of position to roll
            S: Current stock price
            new_T_short: New time to expiration for short leg
            new_sigma_short: New implied vol for new short leg
            close_price_short: Price to close current short leg
            new_price_short: Price to open new short leg

        Returns:
            Roll trade details
        """
        if position_index >= len(self.positions):
            raise ValueError(f"Invalid position index: {position_index}")

        pos = self.positions[position_index]

        # Close old short
        cost_close = close_price_short * pos['quantity'] * 100

        # Open new short
        proceeds_new = new_price_short * pos['quantity'] * 100

        net_credit = proceeds_new - cost_close
        transaction_cost = (cost_close + proceeds_new) * self.transaction_cost

        # Update position
        pos['T_short'] = new_T_short
        pos['sigma_short'] = new_sigma_short
        pos['entry_price_short'] = new_price_short
        pos['net_debit'] -= (net_credit - transaction_cost)

        trade = {
            'action': 'ROLL_SHORT',
            'stock_price': S,
            'strike': pos['K'],
            'close_price': close_price_short,
            'new_price': new_price_short,
            'net_credit': net_credit - transaction_cost,
            'new_T_short': new_T_short
        }

        self.trades.append(trade)
        return trade

    def get_performance_summary(self) -> Dict:
        """
        Get performance summary of all calendar spread trades.

        Returns:
            Dictionary with performance metrics
        """
        closed_trades = [t for t in self.trades if 'pnl' in t]

        if len(closed_trades) == 0:
            return {
                'total_trades': len(self.trades),
                'closed_spreads': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_pnl_per_spread': 0.0
            }

        pnls = [t['pnl'] for t in closed_trades]
        total_pnl = sum(pnls)
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        win_rate = winning_trades / len(closed_trades)

        return {
            'total_trades': len(self.trades),
            'closed_spreads': len(closed_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_pnl_per_spread': total_pnl / len(closed_trades),
            'best_trade': max(pnls),
            'worst_trade': min(pnls),
            'avg_pnl_pct': np.mean([t['pnl_pct'] for t in closed_trades])
        }

    def get_trade_history(self) -> pd.DataFrame:
        """
        Get complete trade history as DataFrame.

        Returns:
            DataFrame with all trades
        """
        return pd.DataFrame(self.trades)
