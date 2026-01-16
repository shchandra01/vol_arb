"""
Straddle and strangle strategies for volatility trading.

Straddle: Buy call and put at the same strike (ATM)
Strangle: Buy call and put at different strikes (OTM)

Both strategies profit from large price movements in either direction
and are used to trade volatility.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ..utils.black_scholes import black_scholes_price, calculate_greeks


class StraddleStrangleStrategy:
    """
    Straddle and strangle strategies for volatility arbitrage.

    Long Straddle/Strangle:
    - Buy when expecting volatility increase
    - Profit from large moves in either direction
    - Benefits from gamma and vega exposure

    Short Straddle/Strangle:
    - Sell when expecting low volatility
    - Profit from time decay if stock stays near strikes
    - Collect premium, but unlimited risk
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,
        profit_target_pct: float = 0.50,
        stop_loss_pct: float = 0.50
    ):
        """
        Initialize straddle/strangle strategy.

        Args:
            transaction_cost: Transaction cost as fraction of trade value
            profit_target_pct: Profit target as percentage of initial cost/credit
            stop_loss_pct: Stop loss as percentage of initial cost/credit
        """
        self.transaction_cost = transaction_cost
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.positions = []
        self.trades = []

    def calculate_straddle_metrics(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> Dict:
        """
        Calculate metrics for an ATM straddle.

        Args:
            S: Current stock price
            K: Strike price (should be near S for ATM)
            T: Time to expiration
            r: Risk-free rate
            sigma: Implied volatility

        Returns:
            Dictionary with straddle metrics
        """
        # Price both options
        call_price = black_scholes_price(S, K, T, r, sigma, 'call')
        put_price = black_scholes_price(S, K, T, r, sigma, 'put')
        total_cost = call_price + put_price

        # Calculate Greeks for both legs
        call_greeks = calculate_greeks(S, K, T, r, sigma, 'call')
        put_greeks = calculate_greeks(S, K, T, r, sigma, 'put')

        # Combined Greeks
        net_delta = call_greeks['delta'] + put_greeks['delta']  # Should be near 0 for ATM
        net_gamma = call_greeks['gamma'] + put_greeks['gamma']
        net_theta = call_greeks['theta'] + put_greeks['theta']
        net_vega = call_greeks['vega'] + put_greeks['vega']

        # Breakeven points
        upper_breakeven = K + total_cost
        lower_breakeven = K - total_cost

        # Maximum profit (for long straddle: unlimited, for short: premium collected)
        # Maximum loss (for long straddle: premium paid, for short: unlimited)

        return {
            'call_price': call_price,
            'put_price': put_price,
            'total_cost': total_cost,
            'net_delta': net_delta,
            'net_gamma': net_gamma,
            'net_theta': net_theta,
            'net_vega': net_vega,
            'upper_breakeven': upper_breakeven,
            'lower_breakeven': lower_breakeven,
            'breakeven_move_pct': (total_cost / K) * 100,
            'call_greeks': call_greeks,
            'put_greeks': put_greeks
        }

    def calculate_strangle_metrics(
        self,
        S: float,
        K_call: float,
        K_put: float,
        T: float,
        r: float,
        sigma: float
    ) -> Dict:
        """
        Calculate metrics for a strangle.

        Args:
            S: Current stock price
            K_call: Call strike (above S)
            K_put: Put strike (below S)
            T: Time to expiration
            r: Risk-free rate
            sigma: Implied volatility

        Returns:
            Dictionary with strangle metrics
        """
        if K_call <= S or K_put >= S:
            print(f"Warning: Strangle strikes may not be OTM. K_call={K_call}, K_put={K_put}, S={S}")

        # Price both options
        call_price = black_scholes_price(S, K_call, T, r, sigma, 'call')
        put_price = black_scholes_price(S, K_put, T, r, sigma, 'put')
        total_cost = call_price + put_price

        # Calculate Greeks for both legs
        call_greeks = calculate_greeks(S, K_call, T, r, sigma, 'call')
        put_greeks = calculate_greeks(S, K_put, T, r, sigma, 'put')

        # Combined Greeks
        net_delta = call_greeks['delta'] + put_greeks['delta']
        net_gamma = call_greeks['gamma'] + put_greeks['gamma']
        net_theta = call_greeks['theta'] + put_greeks['theta']
        net_vega = call_greeks['vega'] + put_greeks['vega']

        # Breakeven points
        upper_breakeven = K_call + total_cost
        lower_breakeven = K_put - total_cost

        return {
            'call_price': call_price,
            'put_price': put_price,
            'total_cost': total_cost,
            'net_delta': net_delta,
            'net_gamma': net_gamma,
            'net_theta': net_theta,
            'net_vega': net_vega,
            'upper_breakeven': upper_breakeven,
            'lower_breakeven': lower_breakeven,
            'breakeven_move_pct_up': ((K_call + total_cost - S) / S) * 100,
            'breakeven_move_pct_down': ((S - K_put + total_cost) / S) * 100,
            'call_greeks': call_greeks,
            'put_greeks': put_greeks
        }

    def enter_long_straddle(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        quantity: int,
        call_price: float,
        put_price: float
    ) -> Dict:
        """
        Enter a long straddle position.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Implied volatility
            quantity: Number of straddles
            call_price: Market price of call
            put_price: Market price of put

        Returns:
            Trade details
        """
        total_cost = (call_price + put_price) * quantity * 100
        transaction_cost = total_cost * self.transaction_cost
        total_paid = total_cost + transaction_cost

        metrics = self.calculate_straddle_metrics(S, K, T, r, sigma)

        position = {
            'type': 'LONG_STRADDLE',
            'S': S,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'quantity': quantity,
            'call_price': call_price,
            'put_price': put_price,
            'total_paid': total_paid,
            'entry_date': None,
            'metrics': metrics
        }

        self.positions.append(position)

        trade = {
            'action': 'ENTER_LONG_STRADDLE',
            'stock_price': S,
            'strike': K,
            'quantity': quantity,
            'call_price': call_price,
            'put_price': put_price,
            'total_cost': total_paid,
            'implied_vol': sigma,
            'net_vega': metrics['net_vega'] * quantity * 100,
            'upper_breakeven': metrics['upper_breakeven'],
            'lower_breakeven': metrics['lower_breakeven']
        }

        self.trades.append(trade)
        return trade

    def enter_short_straddle(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        quantity: int,
        call_price: float,
        put_price: float
    ) -> Dict:
        """
        Enter a short straddle position (high risk).

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Implied volatility
            quantity: Number of straddles
            call_price: Market price of call
            put_price: Market price of put

        Returns:
            Trade details
        """
        total_proceeds = (call_price + put_price) * quantity * 100
        transaction_cost = total_proceeds * self.transaction_cost
        net_proceeds = total_proceeds - transaction_cost

        metrics = self.calculate_straddle_metrics(S, K, T, r, sigma)

        position = {
            'type': 'SHORT_STRADDLE',
            'S': S,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'quantity': quantity,
            'call_price': call_price,
            'put_price': put_price,
            'proceeds': net_proceeds,
            'entry_date': None,
            'metrics': metrics
        }

        self.positions.append(position)

        trade = {
            'action': 'ENTER_SHORT_STRADDLE',
            'stock_price': S,
            'strike': K,
            'quantity': quantity,
            'call_price': call_price,
            'put_price': put_price,
            'proceeds': net_proceeds,
            'implied_vol': sigma,
            'net_vega': -metrics['net_vega'] * quantity * 100,  # Negative for short
            'upper_breakeven': metrics['upper_breakeven'],
            'lower_breakeven': metrics['lower_breakeven']
        }

        self.trades.append(trade)
        return trade

    def enter_long_strangle(
        self,
        S: float,
        K_call: float,
        K_put: float,
        T: float,
        r: float,
        sigma: float,
        quantity: int,
        call_price: float,
        put_price: float
    ) -> Dict:
        """
        Enter a long strangle position.

        Args:
            S: Current stock price
            K_call: Call strike
            K_put: Put strike
            T: Time to expiration
            r: Risk-free rate
            sigma: Implied volatility
            quantity: Number of strangles
            call_price: Market price of call
            put_price: Market price of put

        Returns:
            Trade details
        """
        total_cost = (call_price + put_price) * quantity * 100
        transaction_cost = total_cost * self.transaction_cost
        total_paid = total_cost + transaction_cost

        metrics = self.calculate_strangle_metrics(S, K_call, K_put, T, r, sigma)

        position = {
            'type': 'LONG_STRANGLE',
            'S': S,
            'K_call': K_call,
            'K_put': K_put,
            'T': T,
            'r': r,
            'sigma': sigma,
            'quantity': quantity,
            'call_price': call_price,
            'put_price': put_price,
            'total_paid': total_paid,
            'entry_date': None,
            'metrics': metrics
        }

        self.positions.append(position)

        trade = {
            'action': 'ENTER_LONG_STRANGLE',
            'stock_price': S,
            'call_strike': K_call,
            'put_strike': K_put,
            'quantity': quantity,
            'call_price': call_price,
            'put_price': put_price,
            'total_cost': total_paid,
            'implied_vol': sigma,
            'net_vega': metrics['net_vega'] * quantity * 100,
            'upper_breakeven': metrics['upper_breakeven'],
            'lower_breakeven': metrics['lower_breakeven']
        }

        self.trades.append(trade)
        return trade

    def mark_to_market(
        self,
        position_index: int,
        S: float,
        T: float,
        sigma: float
    ) -> Dict:
        """
        Mark a position to market.

        Args:
            position_index: Index of position to mark
            S: Current stock price
            T: Current time to expiration
            sigma: Current implied volatility

        Returns:
            Current position value and P&L
        """
        if position_index >= len(self.positions):
            raise ValueError(f"Invalid position index: {position_index}")

        pos = self.positions[position_index]

        if pos['type'] in ['LONG_STRADDLE', 'SHORT_STRADDLE']:
            # Straddle
            call_value = black_scholes_price(S, pos['K'], T, pos['r'], sigma, 'call')
            put_value = black_scholes_price(S, pos['K'], T, pos['r'], sigma, 'put')
            total_value = (call_value + put_value) * pos['quantity'] * 100

            if pos['type'] == 'LONG_STRADDLE':
                pnl = total_value - pos['total_paid']
                pnl_pct = (pnl / pos['total_paid']) * 100 if pos['total_paid'] > 0 else 0
            else:  # SHORT_STRADDLE
                pnl = pos['proceeds'] - total_value
                pnl_pct = (pnl / pos['proceeds']) * 100 if pos['proceeds'] > 0 else 0

            greeks = self.calculate_straddle_metrics(S, pos['K'], T, pos['r'], sigma)

        else:  # STRANGLE
            call_value = black_scholes_price(S, pos['K_call'], T, pos['r'], sigma, 'call')
            put_value = black_scholes_price(S, pos['K_put'], T, pos['r'], sigma, 'put')
            total_value = (call_value + put_value) * pos['quantity'] * 100

            if pos['type'] == 'LONG_STRANGLE':
                pnl = total_value - pos['total_paid']
                pnl_pct = (pnl / pos['total_paid']) * 100 if pos['total_paid'] > 0 else 0
            else:  # SHORT_STRANGLE
                pnl = pos['proceeds'] - total_value
                pnl_pct = (pnl / pos['proceeds']) * 100 if pos['proceeds'] > 0 else 0

            greeks = self.calculate_strangle_metrics(S, pos['K_call'], pos['K_put'], T, pos['r'], sigma)

        return {
            'position_index': position_index,
            'position_type': pos['type'],
            'stock_price': S,
            'position_value': total_value,
            'entry_cost': pos.get('total_paid') or pos.get('proceeds'),
            'unrealized_pnl': pnl,
            'pnl_pct': pnl_pct,
            'net_delta': greeks['net_delta'],
            'net_vega': greeks['net_vega'],
            'net_theta': greeks['net_theta'],
            'net_gamma': greeks['net_gamma'],
            'vol_change': sigma - pos['sigma']
        }

    def check_exit_conditions(
        self,
        position_index: int,
        S: float,
        T: float,
        sigma: float
    ) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Args:
            position_index: Index of position to check
            S: Current stock price
            T: Current time to expiration
            sigma: Current implied volatility

        Returns:
            Tuple of (should_exit, reason)
        """
        mtm = self.mark_to_market(position_index, S, T, sigma)

        # Exit condition 1: Profit target reached
        if mtm['pnl_pct'] >= self.profit_target_pct * 100:
            return True, 'PROFIT_TARGET'

        # Exit condition 2: Stop loss hit
        if mtm['pnl_pct'] <= -self.stop_loss_pct * 100:
            return True, 'STOP_LOSS'

        # Exit condition 3: Near expiration
        if T < 7/365:  # Less than 7 days
            return True, 'EXPIRATION'

        return False, 'HOLD'

    def close_position(
        self,
        position_index: int,
        S: float,
        call_price: float,
        put_price: float,
        reason: str = 'MANUAL'
    ) -> Dict:
        """
        Close a straddle/strangle position.

        Args:
            position_index: Index of position to close
            S: Current stock price
            call_price: Current call price
            put_price: Current put price
            reason: Reason for closing

        Returns:
            Trade details
        """
        if position_index >= len(self.positions):
            raise ValueError(f"Invalid position index: {position_index}")

        pos = self.positions[position_index]
        total_value = (call_price + put_price) * pos['quantity'] * 100
        transaction_cost = total_value * self.transaction_cost

        if 'LONG' in pos['type']:
            proceeds = total_value - transaction_cost
            pnl = proceeds - pos['total_paid']
            pnl_pct = (pnl / pos['total_paid']) * 100
        else:  # SHORT
            cost = total_value + transaction_cost
            pnl = pos['proceeds'] - cost
            pnl_pct = (pnl / pos['proceeds']) * 100

        trade = {
            'action': f"CLOSE_{pos['type']}",
            'reason': reason,
            'stock_price': S,
            'entry_stock_price': pos['S'],
            'stock_move': S - pos['S'],
            'stock_move_pct': ((S - pos['S']) / pos['S']) * 100,
            'entry_call_price': pos['call_price'],
            'entry_put_price': pos['put_price'],
            'exit_call_price': call_price,
            'exit_put_price': put_price,
            'entry_vol': pos['sigma'],
            'quantity': pos['quantity'],
            'pnl': pnl,
            'pnl_pct': pnl_pct
        }

        self.trades.append(trade)
        self.positions.pop(position_index)

        return trade

    def get_performance_summary(self) -> Dict:
        """
        Get performance summary of all trades.

        Returns:
            Dictionary with performance metrics
        """
        closed_trades = [t for t in self.trades if 'pnl' in t]

        if len(closed_trades) == 0:
            return {
                'total_trades': len(self.trades),
                'closed_positions': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_pnl_per_trade': 0.0
            }

        pnls = [t['pnl'] for t in closed_trades]
        total_pnl = sum(pnls)
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        win_rate = winning_trades / len(closed_trades)

        return {
            'total_trades': len(self.trades),
            'closed_positions': len(closed_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_pnl_per_trade': total_pnl / len(closed_trades),
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
